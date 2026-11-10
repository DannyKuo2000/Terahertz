"""Configuration overlay for the teacher/student 261106 experiment."""

from config import *


#! [TEACHER-STUDENT-261106] Begin: Isolated experiment configuration
RESTORMER_CONFIG = dict(RESTORMER_CONFIG)
RESTORMER_CONFIG.update({
    "teacher_dim": 24,
    "teacher_num_blocks": [1, 2, 2, 2],
    "teacher_num_refinement_blocks": 2,
    "teacher_loss_weight": 1.0,
    "distillation_loss_weight": 0.05,
})

TRAINING_CONFIG = dict(TRAINING_CONFIG)
TRAINING_CONFIG.update({
    "checkpoints_weights_save_dir": "./checkpoints_weights/Restormer_ONN8LBlock_TS_v1",
    "writer_save_path": "runs/Restormer_ONN8LBlock_TS_v1",
    "resume_training": False,
    "resume_checkpoint_path": "",
})

TESTING_CONFIG = dict(TESTING_CONFIG)
TESTING_CONFIG.update({
    "weight_save_dir": "./checkpoints_weights/Restormer_ONN8LBlock_TS_v1",
    "weight_save_name": "last_checkpoint.pth",
    "results_save_dir": "./results/Restormer_ONN8LBlock_TS_v1",
})
#! [TEACHER-STUDENT-261106] End: Isolated experiment configuration
