import sys
import torch
from arguments import config_parser
import logging
from typing import List

from utils.dataset import to_device_recursive
from utils.visualization import convert_uint
from Avatar import Avatar

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QDesktopWidget, QScrollArea, QComboBox, QLayout, QTableWidget, QGridLayout, QTabWidget
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

def main(avatar: Avatar):
    args = avatar.args
    device, dataset_train, flame, deformer_net, cams_K_train = \
        avatar.device, avatar.dataset_train, avatar.flame, avatar.deformer_net, avatar.cams_K_train

    n_exp = deformer_net.num_exp
    n_joints = flame.n_joints
    expr_labels = avatar.blendshapes_names if args.blendshapes else [str(i) for i in range(n_exp)]
    pose_labels = ["global.x", "global.y", "global.z", "neck.x", "neck.y", "neck.z", "jaw.x", "jaw.y", "jaw.z", "eyeR.x", "eyeR.y", "eyeR.z", "eyeL.x", "eyeL.y", "eyeL.z"]

    view = dataset_train.collate([dataset_train[40]])
    view = to_device_recursive(view, device)

    @torch.no_grad()
    def render_fn(seq_idx: int, expr: torch.Tensor, pose: torch.Tensor):
        view["flame_expression"] = expr
        view["flame_pose"] = pose
        view["seq_idx"][0] = seq_idx
        view["camera"][0].K = cams_K_train[seq_idx]
        rgb, *_ = avatar.run(view, args.resume)
        return rgb
    
    logging.info("Setting up UI")

    exp_min = -1.5
    exp_max = 1.5
    pose_min = -1.0
    pose_max = 1.0

    def widget(layout, parent):
        wid = QWidget(parent)
        wid.setLayout(layout)
        return wid
    def layout_generator(Class):
        def fn(*children):
            l = Class()
            for child in children:
                if isinstance(child, QWidget): l.addWidget(child)
                if isinstance(child, QLayout): l.addLayout(child)
            return l
        return fn
    hbox = layout_generator(QHBoxLayout)
    vbox = layout_generator(QVBoxLayout)
    
    class MainWindow(QMainWindow):
        def __init__(self):
            super(MainWindow, self).__init__()
            self.setWindowTitle(f"MultiFLARE Visualization ({args.run_name} iter {args.resume})")
            self.setGeometry(0, 0, 1100, 512) # (x, y, width, height)
            # Center
            centerPoint = QDesktopWidget().availableGeometry().center()
            qtRectangle = self.frameGeometry()
            qtRectangle.moveCenter(centerPoint)
            self.move(qtRectangle.topLeft())

            self.select_seq_box = QComboBox(self)
            for i in range(len(args.train_dir)):
                self.select_seq_box.addItem(f"Sequence #{i}", i)
            self.select_seq_box.activated.connect(self.update_render)

            layout_expr = QGridLayout()
            layout_expr.setColumnMinimumWidth(1, 200)
            self.exp_sliders: List[QSlider] = []
            self.exp_labels: List[QLabel] = []
            for i in range(n_exp):
                slider = QSlider(Qt.Horizontal, self)
                slider.setRange(0, 100)
                slider.setValue(50)
                label = QLabel()
                slider.valueChanged.connect(self.update_render)
                self.exp_sliders.append(slider)
                self.exp_labels.append(label)
                layout_expr.addWidget(QLabel(f"{expr_labels[i]}: ", self), i, 0)
                layout_expr.addWidget(slider, i, 1)
                layout_expr.addWidget(label, i, 2)

            layout_pose = QGridLayout()
            layout_pose.setColumnMinimumWidth(1, 200)
            self.pose_sliders: List[QSlider] = []
            self.pose_labels: List[QLabel] = []
            for i in range(n_joints*3):
                slider = QSlider(Qt.Horizontal, self)
                slider.setRange(0, 100)
                slider.setValue(50)
                label = QLabel()
                slider.valueChanged.connect(self.update_render)
                self.pose_sliders.append(slider)
                self.pose_labels.append(label)
                layout_pose.addWidget(QLabel(f"{pose_labels[i]}: ", self), i, 0)
                layout_pose.addWidget(slider, i, 1)
                layout_pose.addWidget(label, i, 2)

            self.img_label = QLabel(self)

            tabWidget = QTabWidget(self)

            exp_scroll_area = QScrollArea(self)
            exp_scroll_area.setWidget(widget(layout_expr, self))
            tabWidget.addTab(exp_scroll_area, "Expression")
            
            pose_scroll_area = QScrollArea(self)
            pose_scroll_area.setWidget(widget(layout_pose, self))
            tabWidget.addTab(pose_scroll_area, "Pose")

            root_layout = hbox(
                self.img_label,
                vbox(self.select_seq_box, tabWidget)
            )

            self.setCentralWidget(widget(root_layout, self))

            self.update_render()

        def expr(self):
            values = [exp_min + (exp_max - exp_min) * s.value()/100.0 for s in self.exp_sliders]
            return torch.tensor(values, dtype=torch.float, device=device)

        def pose(self):
            values = [pose_min + (pose_max - pose_min) * s.value()/100.0 for s in self.pose_sliders]
            return torch.tensor(values, dtype=torch.float, device=device)

        def update_render(self):
            expr = self.expr()
            for expr_coeff, l in zip(expr, self.exp_labels):
                v = f"{expr_coeff:.04f}"
                if l.text() != v:
                    l.setText(v)
            pose = self.pose()
            for pose_coeff, l in zip(pose, self.pose_labels):
                v = f"{pose_coeff:.04f}"
                if l.text() != v:
                    l.setText(v)
            seq = self.select_seq_box.currentData()
            img = render_fn(seq, expr.unsqueeze(0), pose.unsqueeze(0)).squeeze(0)
            self.set_image(img)
        
        def set_image(self, img):
            img = convert_uint(img, to_srgb=True)
            height, width, _ = img.shape
            bytesPerLine = 3 * width
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.img_label.setPixmap(QPixmap.fromImage(qImg))

    app = QApplication(sys.argv)
    app.setFont(QFont("Consolas"))
    w = MainWindow()
    w.show()
    app.exec_()


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    #################### Validate args ####################
    if args.resume is None:
        raise ValueError("Arg --resume is required")

    main(Avatar(args))
