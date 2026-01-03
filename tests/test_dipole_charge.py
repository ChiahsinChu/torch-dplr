# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest
from pathlib import Path

import numpy as np
import torch
from deepmd.entrypoints.convert_backend import convert_backend
from deepmd.pt.entrypoints.main import get_trainer
from deepmd.pt.utils.utils import to_numpy_array, to_torch_tensor
from deepmd.tf.modifier import DipoleChargeModifier as TFDipoleChargeModifier
from deepmd.tf.utils.convert import convert_pbtxt_to_pb

from torch_dplr import DipoleChargeModifier as PTDipoleChargeModifier

SEED = 1
DTYPE = torch.float64


def ref_data():
    all_box = np.load(
        str(Path(__file__).parent / "data/energy/data/data_0/set.000/box.npy")
    )
    all_coord = np.load(
        str(Path(__file__).parent / "data/energy/data/data_0/set.000/coord.npy")
    )
    nframe = len(all_box)
    rng = np.random.default_rng(SEED)
    selected_id = rng.integers(nframe)

    coord = all_coord[selected_id].reshape(1, -1)
    box = all_box[selected_id].reshape(1, -1)
    atype = np.loadtxt(
        str(Path(__file__).parent / "data/energy/data/data_0/type.raw"),
        dtype=int,
    ).reshape(1, -1)
    return coord, box, atype


class TestDipoleChargeModifier(unittest.TestCase):
    def setUp(self) -> None:
        # setup parameter
        # numerical consistency can only be achieved with high prec
        self.ewald_h = 0.1
        self.ewald_beta = 0.5
        self.model_charge_map = [-8.0]
        self.sys_charge_map = [6.0, 1.0]

        # Convert pbtxt model to pb model
        convert_pbtxt_to_pb(
            str(Path(__file__).parent / "data/dipole/frozen_model.pbtxt"), "dw_model.pb"
        )
        # Convert pb model to pth model
        convert_backend(INPUT="dw_model.pb", OUTPUT="dw_model.pth")

        self.dm_pt = PTDipoleChargeModifier(
            "dw_model.pth",
            self.model_charge_map,
            self.sys_charge_map,
            self.ewald_h,
            self.ewald_beta,
        )
        self.dm_tf = TFDipoleChargeModifier(
            "dw_model.pb",
            self.model_charge_map,
            self.sys_charge_map,
            self.ewald_h,
            self.ewald_beta,
        )
        print(self.dm_tf.get_sel_type())

    def test_jit(self):
        torch.jit.script(self.dm_pt)

    def test_consistency(self):
        coord, box, atype = ref_data()
        # consistent with the input shape from BaseModifier.modify_data
        t_coord = to_torch_tensor(coord).to(DTYPE).reshape(1, -1, 3)
        t_box = to_torch_tensor(box).to(DTYPE).reshape(1, 3, 3)
        t_atype = to_torch_tensor(atype).to(torch.long)

        dm_pred = self.dm_pt(
            coord=t_coord,
            atype=t_atype,
            box=t_box,
        )
        e, f, v = self.dm_tf.eval(
            coord=coord,
            box=box,
            atype=atype.reshape(-1),
        )

        np.testing.assert_allclose(
            to_numpy_array(dm_pred["energy"]).reshape(-1), e.reshape(-1), rtol=1e-4
        )
        np.testing.assert_allclose(
            to_numpy_array(dm_pred["force"]).reshape(-1), f.reshape(-1), rtol=1e-4
        )
        np.testing.assert_allclose(
            to_numpy_array(dm_pred["virial"]).reshape(-1), v.reshape(-1), rtol=1e-4
        )

    def test_train(self):
        input_json = str(Path(__file__).parent / "data/energy/torch_input.json")
        with open(input_json, encoding="utf-8") as f:
            config = json.load(f)
        config["training"]["save_freq"] = 1
        config["learning_rate"]["start_lr"] = 1.0
        config["training"]["training_data"]["systems"] = str(
            Path(__file__).parent / "data/energy/data"
        )
        config["training"]["numb_steps"] = 10

        trainer = get_trainer(config)
        trainer.run()

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("frozen_model") and f.endswith(".pth"):
                os.remove(f)
            if f.startswith("dw_model") and (f.endswith(".pth") or f.endswith(".pb")):
                os.remove(f)
            if f.startswith("model.ckpt") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out", "checkpoint"]:
                os.remove(f)
