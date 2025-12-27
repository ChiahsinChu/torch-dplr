# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import torch
from deepmd.pt.modifier.base_modifier import BaseModifier
from deepmd.pt.utils import env
from deepmd.pt.utils.utils import to_torch_tensor
from torch_admp.pme import CoulombForceModule
from torch_admp.utils import calc_grads


@BaseModifier.register("dipole_charge")
class DipoleChargeModifier(BaseModifier):
    """Parameters
    ----------
    model_name
            The model file for the DeepDipole model
    model_charge_map
            Gives the amount of charge for the wfcc
    sys_charge_map
            Gives the amount of charge for the real atoms
    ewald_h
            Grid spacing of the reciprocal part of Ewald sum. Unit: A
    ewald_beta
            Splitting parameter of the Ewald sum. Unit: A^{-1}
    """

    def __new__(
        cls, *args: tuple, model_name: str | None = None, **kwargs: dict
    ) -> "DipoleChargeModifier":
        return super().__new__(cls, model_name)

    def __init__(
        self,
        model_name: str,
        model_charge_map: list[float],
        sys_charge_map: list[float],
        ewald_h: float = 1.0,
        ewald_beta: float = 1.0,
    ) -> None:
        """Constructor."""
        super().__init__()
        self.modifier_type = "dipole_charge"
        self.model_name = model_name

        self.model = torch.jit.load(model_name, map_location=env.DEVICE)
        self.rcut = self.model.get_rcut()
        self.type_map = self.model.get_type_map()
        sel_type = self.model.get_sel_type()
        self.sel_type = to_torch_tensor(np.array(sel_type))
        self.model_charge_map = to_torch_tensor(np.array(model_charge_map))
        self.sys_charge_map = to_torch_tensor(np.array(sys_charge_map))

        # init ewald recp
        self.ewald_h = ewald_h
        self.ewald_beta = ewald_beta
        self.er = CoulombForceModule(
            rcut=self.rcut,
            rspace=False,
            kappa=ewald_beta,
            spacing=ewald_h,
        )

        # t_box = to_torch_tensor(box)
        # t_box.requires_grad_(True)
        # frac_positions = positions @ np.linalg.inv(box)
        # t_positions = torch.matmul(to_torch_tensor(frac_positions), t_box)
        # t_charges = to_torch_tensor(charges)

        self.placeholder_pairs = torch.ones((1, 2), device=env.DEVICE, dtype=torch.long)
        self.placeholder_ds = torch.ones((1), device=env.DEVICE, dtype=torch.float64)
        self.placeholder_buffer_scales = torch.zeros(
            (1), device=env.DEVICE, dtype=torch.float64
        )

    def serialize(self) -> dict:
        """Serialize the modifier.

        Returns
        -------
        dict
            The serialized data
        """
        data = {
            "@class": "Modifier",
            "type": self.modifier_type,
            "@version": 3,
            "model_name": self.model_name,
            "model_charge_map": self.model_charge_map,
            "sys_charge_map": self.sys_charge_map,
            "ewald_h": self.ewald_h,
            "ewald_beta": self.ewald_beta,
        }
        return data

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute energy, force, and virial corrections for dipole-charge systems.

        This method extends the system with Wannier Function Charge Centers (WFCC)
        by adding dipole vectors to atomic coordinates for selected atom types.
        It then calculates the electrostatic interactions using Ewald reciprocal
        summation to obtain energy, force, and virial corrections.

        Parameters
        ----------
        coord : torch.Tensor
            The coordinates of atoms with shape (nframes, natoms * 3)
        atype : torch.Tensor
            The atom types with shape (natoms,)
        box : torch.Tensor | None, optional
            The simulation box with shape (nframes, 9), by default None
            Note: This modifier can only be applied for periodic systems
        fparam : torch.Tensor | None, optional
            Frame parameters with shape (nframes, nfp), by default None
        aparam : torch.Tensor | None, optional
            Atom parameters with shape (nframes, natoms, nap), by default None
        do_atomic_virial : bool, optional
            Whether to compute atomic virial, by default False

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the correction terms:
            - energy: Energy correction tensor with shape (nframes, 1)
            - force: Force correction tensor with shape (nframes, natoms+nsel, 3)
            - virial: Virial correction tensor with shape (nframes, 3, 3)
        """
        if box is None:
            raise RuntimeWarning(
                "dipole_charge data modifier can only be applied for periodic systems."
            )
        else:
            modifier_pred = {}
            nframes = coord.shape[0]
            natoms = coord.shape[1] // 3

            extended_coord, extended_charge, atomic_dipole = self.extend_system(
                coord,
                atype,
                box,
                fparam,
                aparam,
            )

            tot_e = []
            all_f = []
            all_v = []

            # add Ewald reciprocal correction
            for ii in range(nframes):
                e, f, v = self.er_eval(extended_coord[ii], box[ii], extended_charge[ii])
                tot_e.append(e.unsqueeze(0))
                all_f.append(f.unsqueeze(0))
                all_v.append(v.unsqueeze(0))
            # nframe,
            tot_e = torch.concat(tot_e, dim=0)
            # nframe, nat + sel, 3
            all_f = torch.concat(all_f, dim=0)
            # nframe, 3,  3
            all_v = torch.concat(all_v, dim=0)

            # electrostatic forces on WC
            # nframe, sel, 3
            ext_f = all_f[:, natoms:, :]
            # nframe, natoms
            mask = make_mask(self.sel_type, atype)
            # map ext_f back to nat length
            ext_f_mapped = torch.zeros(
                nframes, natoms, 3, device=coord.device, dtype=coord.dtype
            )
            for ii in range(nframes):
                ext_f_mapped[ii][mask[ii]] = ext_f[ii]

            corr_f = []
            corr_v = []
            for ii in range(nframes):
                output = self.model(
                    coord=coord[ii].reshape(1, -1),
                    atype=atype.reshape(1, -1),
                    box=box[ii].reshape(1, -1),
                    do_atomic_virial=False,
                    fparam=fparam[ii].reshape(1, -1) if fparam is not None else None,
                    aparam=aparam[ii].reshape(1, -1) if aparam is not None else None,
                    atomic_weight=ext_f_mapped[ii].reshape(1, -1),
                )
                corr_f.append(-output["force"])
                corr_v.append(-output["virial"].reshape(1, -1, 9))
            # nframes, natoms, nout, 3
            corr_f = torch.concat(corr_f, dim=0)
            # nframes, natoms, 3
            corr_f = torch.sum(corr_f, dim=2)
            # nframes, nout, 9
            corr_v = torch.concat(corr_v, dim=0)
            # nframe, 9
            corr_v = torch.sum(corr_v, dim=1)
            # print(corr_f.shape, corr_v.shape)

            # compute f
            # nframe, natoms, 3
            tot_f = all_f[:, :natoms, :] + ext_f_mapped + corr_f
            # compute v
            ext_f3 = ext_f_mapped.permute(0, 2, 1)
            # nframe, 3,  3
            fd_corr_v = -torch.matmul(ext_f3, atomic_dipole)
            tot_v = all_v + corr_v.reshape(-1, 3, 3) + fd_corr_v

            modifier_pred["energy"] = tot_e
            modifier_pred["force"] = tot_f
            modifier_pred["virial"] = tot_v
            return modifier_pred

    def extend_system(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extend the system with WFCC (Wannier Function Charge Centers).

        Parameters
        ----------
        coord : torch.Tensor
            The coordinates of atoms with shape (nframes, natoms * 3)
        atype : torch.Tensor
            The atom types with shape (natoms,)
        box : torch.Tensor
            The simulation box with shape (nframes, 9)
        fparam : torch.Tensor | None, optional
            Frame parameters with shape (nframes, nfp), by default None
        aparam : torch.Tensor | None, optional
            Atom parameters with shape (nframes, natoms, nap), by default None

        Returns
        -------
        tuple
            (extended_coord, extended_charge, atomic_dipole)
            extended_coord : torch.Tensor
                Extended coordinates with shape (nframes, (natoms + nsel) * 3)
            extended_charge : torch.Tensor
                Extended charges with shape (nframes, natoms + nsel)
            atomic_dipole : torch.Tensor
                Dipole values with shape (nframes, natoms, 3)
        """
        nframes = coord.shape[0]
        mask = make_mask(self.sel_type, atype)

        extended_coord, atomic_dipole = self.extend_system_coord(
            coord,
            atype,
            box,
            fparam,
            aparam,
        )
        # Get ion charges based on atom types
        # nframe x nat
        ion_charge = self.sys_charge_map[atype]
        # Initialize wfcc charges
        wc_charge = torch.zeros_like(ion_charge)
        # Assign charges to selected atom types
        for ii, charge in enumerate(self.model_charge_map):
            wc_charge[atype == self.sel_type[ii]] = charge
        # Get the charges for selected atoms only
        wc_charge_selected = wc_charge[mask].reshape(nframes, -1)
        # Concatenate ion charges and wfcc charges
        extended_charge = torch.cat([ion_charge, wc_charge_selected], dim=1)
        return extended_coord, extended_charge, atomic_dipole

    def extend_system_coord(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extend the system with WFCC (Wannier Function Charge Centers).

        This function calculates Wannier Function Charge Centers (WFCC) by adding dipole
        vectors to atomic coordinates for selected atom types, then concatenates these
        WFCC coordinates with the original atomic coordinates.

        Parameters
        ----------
        coord : torch.Tensor
            The coordinates of atoms with shape (nframes, natoms * 3)
        atype : torch.Tensor
            The atom types with shape (natoms,)
        box : torch.Tensor
            The simulation box with shape (nframes, 9)
        fparam : torch.Tensor | None, optional
            Frame parameters with shape (nframes, nfp), by default None
        aparam : torch.Tensor | None, optional
            Atom parameters with shape (nframes, natoms, nap), by default None

        Returns
        -------
        tuple
            (all_coord, dipole) - extended coordinates including WFCC and the dipole values
            all_coord : torch.Tensor
                Extended coordinates with shape (nframes, (natoms + nsel) * 3)
                where nsel is the number of selected atoms
            dipole : torch.Tensor
                Dipole values with shape (nframes, natoms * 3)
        """
        mask = make_mask(self.sel_type, atype)

        nframes = coord.shape[0]
        natoms = coord.shape[1] // 3

        all_dipole = []
        for ii in range(nframes):
            dipole_batch = self.model(
                coord=coord[ii].reshape(1, -1),
                atype=atype.reshape(1, -1),
                box=box[ii].reshape(1, -1),
                do_atomic_virial=False,
                fparam=fparam[ii].reshape(1, -1) if fparam is not None else None,
                aparam=aparam[ii].reshape(1, -1) if aparam is not None else None,
            )
            # Extract dipole from the output dictionary
            all_dipole.append(dipole_batch["dipole"])

        # nframe x natoms x 3
        dipole = torch.cat(all_dipole, dim=0)
        assert dipole.shape[0] == nframes

        dipole_reshaped = dipole.reshape(nframes, natoms, 3)
        coord_reshaped = coord.reshape(nframes, natoms, 3)
        _wfcc_coord = coord_reshaped + dipole_reshaped
        # Apply mask and reshape
        wfcc_coord = _wfcc_coord[mask.unsqueeze(-1).expand_as(_wfcc_coord)]
        wfcc_coord = wfcc_coord.reshape(nframes, -1)
        all_coord = torch.cat((coord, wfcc_coord), dim=1)
        return all_coord, dipole

    def er_eval(
        self,
        _coord: torch.Tensor,
        _box: torch.Tensor,
        _charge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        box = torch.reshape(_box, (3, 3))
        box.requires_grad_(True)
        frac_positions = torch.matmul(
            _coord.reshape(-1, 3),
            torch.linalg.inv(box),
        )
        detached_frac_positions = frac_positions.detach()
        positions = torch.matmul(detached_frac_positions, box)
        charges = torch.reshape(_charge, (-1,))

        self.er(
            positions,
            box,
            self.placeholder_pairs,
            self.placeholder_ds,
            self.placeholder_buffer_scales,
            {"charge": charges},
        )
        e = self.er.reciprocal_energy
        f = -calc_grads(e, positions)
        v = calc_grads(e, box)
        v = -torch.matmul(v.transpose(1, 0), box)
        return e, f, v


@torch.jit.export
def make_mask(
    sel_type: torch.Tensor,
    atype: torch.Tensor,
) -> torch.Tensor:
    """Create a boolean mask for selected atom types.

    Parameters
    ----------
    sel_type : torch.Tensor
        The selected atom types to create a mask for
    atype : torch.Tensor
        The atom types in the system

    Returns
    -------
    mask : torch.Tensor
        Boolean mask where True indicates atoms of selected types
    """
    # Ensure tensors are of the right type
    sel_type = sel_type.to(torch.long)
    atype = atype.to(torch.long)

    # Create mask using broadcasting
    mask = torch.zeros_like(atype, dtype=torch.bool)
    for t in sel_type:
        mask = mask | (atype == t)
    return mask
