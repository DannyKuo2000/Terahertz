"""Teacher/student Restormer variant for the 2026-11-06 experiment.

The student keeps the existing RestormerOfficial interface.  The teacher is
smaller and is used only when a training-time optical field is supplied.
"""

import copy

import torch
import torch.nn as nn

from model.RestormerOfficial import Restormer as OfficialRestormer


class Restormer(nn.Module):
    #! [TEACHER-STUDENT-261106] Begin: Student/teacher wrapper
    def __init__(self, config):
        super(Restormer, self).__init__()
        self.student = OfficialRestormer(config)

        teacher_config = copy.deepcopy(config)
        teacher_config["dim"] = config.get("teacher_dim", max(16, config["dim"] // 2))
        teacher_config["num_blocks"] = config.get(
            "teacher_num_blocks", [1, 2, 2, 2]
        )
        teacher_config["heads"] = config.get(
            "teacher_heads", [1, 2, 4, 8]
        )
        teacher_config["num_refinement_blocks"] = config.get(
            "teacher_num_refinement_blocks", 2
        )
        #! [TEACHER-STUDENT-261106] Teacher directly consumes
        # amplitude/sin(phase)/cos(phase), avoiding a 3->1 information bottleneck.
        teacher_config["inp_channels"] = 3
        teacher_config["out_channels"] = config["out_channels"]
        # Kept as a named module for checkpoint/interface compatibility.
        self.teacher_input_projection = nn.Identity()
        self.teacher = OfficialRestormer(teacher_config)
        # Official Restormer adds the input tensor as a residual. With a
        # 3-channel teacher input this produces 3 output channels, so project
        # the teacher prediction back to the reconstruction channel count.
        self.teacher_output_projection = nn.Conv2d(
            3, config["out_channels"], kernel_size=1
        )
        #! [TEACHER-STUDENT-261106] Added: Intermediate feature capture and alignment
        feature_names = (
            "encoder_level1", "encoder_level2", "encoder_level3", "latent",
            "decoder_level3", "decoder_level2", "decoder_level1", "refinement",
        )
        self._student_features = {}
        self._teacher_features = {}
        self._feature_hooks = []
        for name in feature_names:
            self._feature_hooks.append(
                getattr(self.student, name).register_forward_hook(
                    self._make_feature_hook(self._student_features, name)
                )
            )
            self._feature_hooks.append(
                getattr(self.teacher, name).register_forward_hook(
                    self._make_feature_hook(self._teacher_features, name)
                )
            )
        student_dim = config["dim"]
        teacher_dim = teacher_config["dim"]
        feature_multipliers = {
            "encoder_level1": 1, "encoder_level2": 2, "encoder_level3": 4,
            "latent": 8, "decoder_level3": 4, "decoder_level2": 2,
            "decoder_level1": 2, "refinement": 2,
        }
        self.teacher_feature_projections = nn.ModuleDict({
            name: nn.Conv2d(
                teacher_dim * multiplier,
                student_dim * multiplier,
                kernel_size=1,
            )
            for name, multiplier in feature_multipliers.items()
        })

    @staticmethod
    def _make_feature_hook(storage, name):
        def hook(_module, _inputs, output):
            storage[name] = output
        return hook

    def forward(self, x, teacher_input=None, return_teacher=False):
        student_output = self.student(x)
        if teacher_input is None or not return_teacher:
            return student_output

        teacher_input = self.teacher_input_projection(teacher_input)
        teacher_output = self.teacher(teacher_input)
        teacher_output = self.teacher_output_projection(teacher_output)
        return student_output, teacher_output

    def forward_teacher_student(self, student_input, teacher_input):
        #! [TEACHER-STUDENT-261106] Added: Joint output and feature forward
        self._student_features.clear()
        self._teacher_features.clear()
        student_output = self.student(student_input)
        teacher_output = self.teacher(self.teacher_input_projection(teacher_input))
        teacher_output = self.teacher_output_projection(teacher_output)
        return (
            student_output,
            teacher_output,
            dict(self._student_features),
            dict(self._teacher_features),
        )

    #! [TEACHER-STUDENT-261106] End: Student/teacher wrapper

    def student_state_dict(self):
        return self.student.state_dict()

    def load_student_state_dict(self, state_dict, strict=True):
        return self.student.load_state_dict(state_dict, strict=strict)
