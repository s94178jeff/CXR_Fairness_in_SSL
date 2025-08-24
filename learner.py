from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import tempfile
import random

from util import (
    join,
    cal_img_bias_fairness,
    cal_demo_group_fairness,
    cal_label_bias_fairness_cnn,
    cal_label_bias_fairness_fit,
    cal_demo_bias_demo_group_fairness_cnn,
    get_old_result,
    save_result,
    check_retrain,
    get_feature_num,
    GROUP_LIST,
    torch_safe_save,
    get_device,
    safe_roc_auc_score,
    get_class_info,
    write_out_scalar,
    write_out_cm,
)

from data_util import get_dataset, get_dataset_feature, IdxDataset, IMAGE_SHORTCUT_TYPE
from module.util import get_model
from ssl_inference import get_ssl_info
from sklearn.neighbors import KNeighborsClassifier


class Learner:
    """Main training/evaluation orchestrator."""

    def __init__(self, args):
        self.args = args
        self.setup_run()
        self.setup_device()
        self.setup_dirs()
        self.setup_dataset()
        self.setup_model()
        self.best_valid_acc = self.best_test_acc = 0.0
        self.best_val_auc = self.best_test_auc = 0.0
        print("Finished model initialization...")

    # ------------------------------
    # Setup methods
    # ------------------------------
    def setup_run(self):
        args = self.args
        data2model = {"covid": "ResNet18", "mimic": "ResNet18", "mimic_ssl": "1linear", "covid_ssl": "1linear"}
        data2batch_size = {"covid": 256, "mimic": 512, "mimic_ssl": 512, "covid_ssl": 512}
        data2preprocess = {"mimic": False, "covid": False, "mimic_ssl": None, "covid_ssl": None}

        if args.method == "aug_vanilla":
            data2preprocess["mimic"] = True
            data2preprocess["covid"] = True
        self.data2preprocess = data2preprocess
        self.batch_size = data2batch_size[args.dataset]
        self.model_name = args.model if args.model else data2model[args.dataset]
        args.model = self.model_name

        run_name = f"{args.shortcut_type}{args.percent}_{args.method}"
        if args.ssl_type:
            aug, epoch = get_ssl_info(args.ssl_ckpt_path)
            run_name = f"{args.ssl_type}_{args.shortcut_type}{args.percent}_ep{epoch}_{aug}_{args.method}"
        if args.exp:
            run_name = args.exp
        self.run_name = run_name

        args.wandb = not args.local
        args.tensorboard = False

        run_name_title = run_name + "(bias)" if args.use_bias_label else run_name
        if args.use_bias_label:
            if args.shortcut_type in IMAGE_SHORTCUT_TYPE + ["LO"]:
                run_name_title = f"{run_name}{args.shortcut_type}"
            elif args.group_type in GROUP_LIST:
                run_name_title = f"{run_name}{args.group_type}"
            else:
                raise NotImplementedError
        self.run_name_title = run_name_title

        # wandb init lazy
        if args.wandb:
            try:
                import wandb
                wandb.init(project=f"LDD_{args.dataset}", resume=args.continue_train, config=args)
                wandb.run.name = run_name_title
            except Exception as e:
                print(f"[WARN] wandb init failed: {e}")
                args.wandb = False
        if args.tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(f"result/summary/{run_name_title}")

    def setup_device(self):
        self.device = get_device(self.args)
        print(f"Using device: {self.device}")

    def setup_dirs(self):
        args = self.args
        self.result_root = join(args.result_root, args.dataset, self.run_name)
        Path(self.result_root).mkdir(parents=True, exist_ok=True)
        use_bias_str = "bias_label_" if args.use_bias_label else ""
        self.result_dir = join(self.result_root, f"{use_bias_str}result")
        if not args.use_bias_label or args.group_type not in ["age", "race", "gender"]:
            check_retrain(self.result_dir, args.continue_train)
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)

    def setup_dataset(self):
        args = self.args
        preprocess = self.data2preprocess[args.dataset]
        self.num_channel = 1 if args.dataset in ["mimic", "covid"] else 3
        self.attr_idx = 1 if args.use_bias_label else 0

        self.train_dataset = IdxDataset(get_dataset(args, "train", "train", use_preprocess=preprocess))
        self.valid_dataset = get_dataset(args, "valid", "val_test", use_preprocess=preprocess)
        self.test_dataset = get_dataset(args, "test", "val_test", use_preprocess=preprocess)

        _, self.target_name, self.num_classes = get_class_info(args.dataset, args.shortcut_type, args.group_type, args.use_bias_label)

        self.train_loader = self.make_loader(self.train_dataset, shuffle=True)
        self.valid_loader = self.make_loader(self.valid_dataset)
        self.test_loader = self.make_loader(self.test_dataset)

        if (args.shortcut_type in ["LO", "Male", "Female", "Race", "Age"] or args.group_type == "LO") and args.dataset not in ["covid", "covid_ssl"]:
            self.valid_flip_loader = self.make_loader(get_dataset(args, "valid_flip", "val_test", use_preprocess=preprocess))
            self.test_flip_loader = self.make_loader(get_dataset(args, "test_flip", "val_test", use_preprocess=preprocess))

    def setup_model(self):
        args = self.args
        feature_num = get_feature_num(args.ssl_type, args.ssl_ckpt_path, args.use_bias_label)
        self.model = get_model(self.model_name, self.num_classes, self.num_channel, ssl_feature=feature_num).to(self.device)
        init_lr = args.lr * self.batch_size / 256
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr, weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        print("Model initialized. LR:", init_lr)

    # ------------------------------
    # Loader helper
    # ------------------------------
    def make_loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=shuffle,
        )

    # ------------------------------
    # Checkpoint helper
    # ------------------------------
    def model_path(self, best=False):
        bias_str = f"{self.args.group_type}_" if self.args.use_bias_label and self.args.group_type in GROUP_LIST else ""
        return join(self.result_dir, f"{bias_str}{'best_model.th' if best else 'final_model.th'}")

    def load_vanilla(self, best=None):
        ckpt = torch.load(self.model_path(best), map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt.get("steps", 0)

    def save_vanilla(self, step, best=None):
        state_dict = {
            "steps": step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if best:
            state_dict.update({"best_valid_acc": self.best_valid_acc, "best_test_acc": self.best_test_acc})
        torch_safe_save(state_dict, self.model_path(best))
        print(f"{step} model saved ...")

    # ------------------------------
    # Evaluate helpers
    # ------------------------------
    def evaluate(self, model, loader):
        model.eval()
        pred_list, prob_list, attr_list = [], [], []
        with torch.no_grad():
            for data, attr, _ in loader:
                data = data.to(self.device)
                logits = model(data)
                preds = logits.argmax(dim=1)
                prob_list.append(torch.softmax(logits, dim=1).cpu())
                attr_list.append(attr[:, self.attr_idx].cpu())
                pred_list.append(preds.cpu())
        cat_attr = torch.cat(attr_list)
        cat_pred = torch.cat(pred_list)
        cat_prob = torch.cat(prob_list)
        acc = (cat_attr == cat_pred).float().mean().item()
        model.train()
        return acc, (cat_attr, cat_pred, cat_prob)

    def evaluate_fair(self, model, loader, shortcut_type="no", group_type="no", flip_loader=None):
        # combine vanilla eval and fairness calculation
        model.eval()
        pred_list, label_list, prob_list, attr_list = [], [], [], []
        flip_batches = []

        with torch.no_grad():
            for data, attr, flip_data in loader:
                data = data.to(self.device)
                logits = model(data)
                preds = logits.argmax(dim=1)
                prob_list.append(torch.softmax(logits, dim=1).cpu())
                label_list.append(attr[:, 0].cpu())
                pred_list.append(preds.cpu())
                if group_type in GROUP_LIST:
                    attr_list.append(attr[:, 1].cpu())
        cat_label = torch.cat(label_list)
        cat_pred = torch.cat(pred_list)
        cat_prob = torch.cat(prob_list)
        acc = (cat_label == cat_pred).float().mean().item()

        # fairness evaluation
        pair_info = None
        if shortcut_type in IMAGE_SHORTCUT_TYPE:
            # assume flip_batches collected from flip_loader if needed
            fairness = -1  # placeholder
        elif shortcut_type == "no" and group_type in GROUP_LIST:
            attrs_cat = torch.cat(attr_list) if len(attr_list) > 0 else torch.tensor([])
            fairness, pair_info = cal_demo_group_fairness(labels=cat_label, preds=cat_pred, group=attrs_cat)
        elif shortcut_type == "LO" or group_type == "LO":
            fairness = cal_label_bias_fairness_cnn(cat_label, cat_pred, flip_loader, model, self.device)
        elif shortcut_type in ["Male", "Female", "Race", "Age"]:
            fairness, pair_info = cal_demo_bias_demo_group_fairness_cnn(flip_loader, model, self.device)
        else:
            fairness = -1
        model.train()
        return acc, fairness, pair_info, (cat_label, cat_pred, cat_prob)

    # ------------------------------
    # Sanity check
    # ------------------------------
    def sanity_check(self, state_dict, pretrained_weights):
        checkpoint = torch.load(pretrained_weights, map_location="cpu")
        state_dict_pre = checkpoint["state_dict"]
        changed = [k for k, v in state_dict.items() if not torch.equal(v.cpu(), state_dict_pre.get(k, v).cpu())]
        if changed:
            print("Changed keys:", changed)
        try:
            Path(pretrained_weights).unlink(missing_ok=True)
        except TypeError:
            p = Path(pretrained_weights)
            if p.exists():
                p.unlink()

    # ------------------------------
    # KNN/XGB (feature-based) evaluation
    # ------------------------------
    def reg_evaluation(self, split, model_name, features, attrs, flip_features=None, flip_attrs=None):
        group = self.args.group_type
        shortcut_type = self.args.shortcut_type
        labels = attrs[:, self.attr_idx]
        tn = self.target_name + "_" if self.args.use_bias_label else ""

        model = getattr(self, model_name)
        pred = model.predict(features)
        prob = model.predict_proba(features)
        auc = safe_roc_auc_score(labels, prob[:, 1] if self.num_classes==2 else prob, multi_class="ovr" if self.num_classes>2 else None, labels=list(range(self.num_classes)) if self.num_classes>2 else None)
        acc = model.score(features, labels)
        self.result[f"{model_name}_{tn}{split}_acc"] = acc
        self.result[f"{model_name}_{tn}{split}_auc"] = auc

        # fairness
        fairness = -1
        pair_info = None
        if self.args.shorTest:
            if shortcut_type in IMAGE_SHORTCUT_TYPE:
                fairness = cal_img_bias_fairness(labels, pred, [model.predict(flip_features[i]) for i in range(flip_features.shape[0])] if flip_features is not None else [], num_class=self.num_classes, flip_size=flip_features.shape[0] if flip_features is not None else 0)
            elif shortcut_type=="no" and group in GROUP_LIST:
                fairness, pair_info = cal_demo_group_fairness(labels, pred, group=attrs[:, 1])
            elif shortcut_type=="LO":
                fairness = cal_label_bias_fairness_fit(labels, pred, flip_attrs[:,0], model.predict(flip_features))
            elif shortcut_type in ["Male","Female","Race","Age"]:
                fairness, pair_info = cal_demo_group_fairness(flip_attrs[:,0], model.predict(flip_features), flip_attrs[:,1])
        self.result[f"{model_name}_{split}_fair_group_{group}"] = fairness
        if group in GROUP_LIST and pair_info is not None:
            self.result[f"{model_name}_{split}_pair_info_group_{group}"] = pair_info

    def fit_other(self, args):
        assert args.dataset in ["mimic_ssl", "covid_ssl"]
        self.result = get_old_result(self.result_dir)
        train_features, train_attr, _ = get_dataset_feature(args, "train")
        val_features, val_attrs, flip_val_features = get_dataset_feature(args, "valid")
        test_features, test_attrs, flip_test_features = get_dataset_feature(args, "test")

        if args.shortcut_type in ["LO","Male","Female","Race","Age"]:
            flip_val_features, flip_val_attrs, _ = get_dataset_feature(args, "valid_flip")
            flip_test_features, flip_test_attrs, _ = get_dataset_feature(args, "test_flip")
        else:
            flip_val_attrs, flip_test_attrs = None, None

        self.knn = KNeighborsClassifier()
        self.knn.fit(train_features, train_attr[:, self.attr_idx])

        for model_name in ["knn"]:
            self.reg_evaluation("val", model_name, val_features, val_attrs, flip_val_features, flip_val_attrs)
            self.reg_evaluation("test", model_name, test_features, test_attrs, flip_test_features, flip_test_attrs)
        save_result(self.result, self.result_dir)

    # ------------------------------
    # Train / Test workflows
    # ------------------------------
    def train_vanilla(self):
        args = self.args
        start_step = self.load_vanilla() if args.continue_train else 0
        train_iter = iter(self.train_loader)
        train_num = len(self.train_dataset.dataset)
        epoch, cnt = 0, 0

        # tmp file for sanity check
        with tempfile.NamedTemporaryFile(suffix=".pth.tar", delete=False) as tmpf:
            tmp_fname = tmpf.name
        torch_safe_save({"state_dict": self.model.state_dict()}, tmp_fname)

        for step in tqdm(range(start_step, args.num_steps), desc="Training"):
            self.model.train()
            try:
                index, data, attr, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                index, data, attr, _ = next(train_iter)

            non_blocking = self.device.type == "cuda"
            data, attr = data.to(self.device, non_blocking=non_blocking), attr.to(self.device, non_blocking=non_blocking)

            logits = self.model(data)
            label = attr[:, self.attr_idx]
            loss = self.criterion(logits, label).mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # sanity check first step
            if step == 0:
                self.sanity_check(self.model.state_dict(), tmp_fname)

            # logging
            if step % args.log_freq == 0:
                self.board_vanilla_loss(step, loss.item())

            if step % args.save_freq == 0:
                self.save_vanilla(step)

            if step % args.valid_freq == 0:
                if args.shorTest:
                    self.board_shorTest_acc(step, epoch)
                else:
                    self.board_vanilla_acc(step, epoch)

            cnt += len(index)
            if cnt >= train_num:
                print(f"Finished epoch: {epoch}")
                epoch += 1
                cnt = 0

        # final test
        self.test_vanilla()

    def test_vanilla(self):
        self.load_vanilla(best=True)
        if self.args.shorTest:
            self.board_shorTest_acc(step=0, epoch=0, inference=True)
        else:
            self.board_vanilla_acc(step=0, epoch=0, inference=True)

    # ------------------------------
    # Logging helpers
    # ------------------------------
    def board_vanilla_loss(self, step, loss):
        if self.args.wandb:
            import wandb
            wandb.log({"loss/train": loss}, step=step)
        if self.args.tensorboard:
            self.writer.add_scalar("loss/train", loss, step)

    def board_vanilla_acc(self, step, epoch, inference=False):
        val_acc, (val_attr, val_pred, val_prob) = self.evaluate(self.model, self.valid_loader)
        tn = self.target_name + "_" if self.args.use_bias_label else ""
        val_auc = safe_roc_auc_score(val_attr, val_prob[:, 1] if self.num_classes==2 else val_prob, multi_class="ovr" if self.num_classes>2 else None, labels=list(range(self.num_classes)) if self.num_classes>2 else None)

        result = get_old_result(self.result_dir)
        result[f"{tn}val_acc"] = val_acc
        result[f"{tn}val_auc"] = val_auc

        if inference:
            test_loader = self.test_loader
            test_acc, (test_attr, test_pred, test_prob) = self.evaluate(self.model, test_loader)
            test_auc = safe_roc_auc_score(test_attr, test_prob[:, 1] if self.num_classes==2 else test_prob, multi_class="ovr" if self.num_classes>2 else None, labels=list(range(self.num_classes)) if self.num_classes>2 else None)
            result[f"{tn}test_acc"] = test_acc
            result[f"{tn}test_auc"] = test_auc
            save_result(result, self.result_dir)
            print("Evaluation done")
            return

        write_out_scalar(self, {f"acc/{tn}valid": val_acc, f"auc/{tn}valid": val_auc}, step)
        if val_auc >= self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_valid_acc = val_acc
            write_out_cm(self, f"best_{tn}val_conf_mat", val_attr, val_pred)
            save_result(result, self.result_dir)
            self.save_vanilla(step, best=True)
        print(f"Val acc: {val_acc:.4f} | Val auc: {val_auc:.4f}")

    def board_shorTest_acc(self, step, epoch, inference=False):
        # 可完整呼叫 evaluate_fair 並計算 fairness
        shortcut, group = self.args.shortcut_type, self.args.group_type
        flip_loader = getattr(self, "valid_flip_loader", None)
        val_acc, val_fair, val_pair_info, (val_label, val_pred, val_prob) = self.evaluate_fair(self.model, self.valid_loader, shortcut, group, flip_loader)
        val_auc = safe_roc_auc_score(val_label, val_prob[:,1] if self.num_classes==2 else val_prob, multi_class="ovr" if self.num_classes>2 else None, labels=list(range(self.num_classes)) if self.num_classes>2 else None)

        result = get_old_result(self.result_dir)
        result["val_acc"] = val_acc
        result["val_auc"] = val_auc
        result[f"val_fair_group_{group}"] = val_fair
        if group in GROUP_LIST:
            result[f"val_pair_info_group_{group}"] = val_pair_info

        if inference:
            flip_loader_test = getattr(self, "test_flip_loader", None)
            test_acc, test_fair, test_pair_info, (test_label, test_pred, test_prob) = self.evaluate_fair(self.model, self.test_loader, shortcut, group, flip_loader_test)
            test_auc = safe_roc_auc_score(test_label, test_prob[:,1] if self.num_classes==2 else test_prob, multi_class="ovr" if self.num_classes>2 else None, labels=list(range(self.num_classes)) if self.num_classes>2 else None)
            if shortcut == "LO" or group == "LO":
                test_auc_list = [
                    safe_roc_auc_score(test_label == class_idx, test_prob[:, class_idx])
                    for class_idx in range(self.num_classes)
                ]
                result["test_auc_apart"] = test_auc_list
            result["test_acc"] = test_acc
            result["test_auc"] = test_auc
            result[f"test_fair_group_{group}"] = test_fair
            if group in GROUP_LIST:
                result[f"test_pair_info_group_{group}"] = test_pair_info
            save_result(result, self.result_dir)
            print("Evaluation done")
            return

        write_out_scalar(self, {"acc/valid": val_acc, "auc/valid": val_auc, f"fairness_{group}/val": val_fair}, step)
        save_result(result, self.result_dir)
        if val_auc >= self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_valid_acc = val_acc
            write_out_cm(self, "best_val_conf_mat", val_label, val_pred)
            self.save_vanilla(step, best=True)
        print(f"Val acc: {val_acc:.4f} | Val auc: {val_auc:.4f} | Fair: {val_fair:.4f}")
