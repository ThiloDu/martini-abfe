# martini-abfe

Absolute protein-ligand binding free energies with the Martini 3 coarse-grained force field, for arbitrary protein-ligand systems.

This is a generalized version of the protocol in [martini-protein-ligand-binding-free-energies](../martini-protein-ligand-binding-free-energies) (package `cgbfe`). That repository is tied to the OpenFE Industry Benchmark 2024. This one takes any system as input and leaves out the ligand parametrization. The simulation protocol, lambda schedules, restraints and free energy estimators are unchanged. The simulation boxes of both legs are built with [COBY](https://github.com/MikkelDA/COBY) instead of insane and packmol.

## Input

| Input              | Description                                                                         |
| ------------------ | ----------------------------------------------------------------------------------- |
| complex `.pdb`     | Atomistic structure of the ligand bound to the protein, e.g. a crystal structure    |
| ligand residue     | Residue name of the ligand in the complex                                           |
| ligand `.itp`      | Martini 3 topology of the ligand (a single molecule with explicit charges)          |
| ligand `.gro`      | Coarse-grained structure of the ligand, beads in topology order                     |
| `mapping.ndx`      | *Optional:* bead mapping, used to superimpose the CG ligand onto the crystal pose   |

The ligand topology can use any molecule and residue names. They are renamed to `ligand` and `LIG`, the names that the mdp files and index groups refer to. Atom names in the structure are taken from the topology.

## Installation

```bash
git clone --recurse-submodules <repository>
pip install -e .
```

`pip` installs martinize2 (vermouth) and COBY, which builds the solvated boxes of both legs. In addition, you need GROMACS built with MPI, with `gmx_mpi` on your path. It is used for all GROMACS commands, including replica exchange. Set thread counts with `-ntomp`, since `-nt` only exists in thread-MPI builds. An explicit `-ntomp` takes precedence over `OMP_NUM_THREADS`, which is removed from the environment of these commands.

The Martini 3 force field is not included. By default, the topologies include `martini3.ff/martini.itp`, `martini3.ff/martini_solvents.itp` and `martini3.ff/martini_ions.itp`, which `grompp` must find through `GMXLIB`. Alternatively, set absolute paths in the config:

```yaml
simulation:
  martini:
    topology: /path/to/martini_v3.0.0.itp
    solvents: /path/to/martini_v3.0.0_solvents_v1.itp
    ions: /path/to/martini_v3.0.0_ions_v1.itp
```

The force field has to contain the bead types used by your ligand, and the molecules `W`, `NA` and `CL`.

The protein tertiary structure is kept with [OLIVES](https://github.com/Martini-Force-Field-Initiative/OLIVES), which is included as a submodule in `external/OLIVES`. It is found relative to the source tree, so install the package in editable mode (`pip install -e .`). With a regular install, or to use another script, set the environment variable `MARTINI_ABFE_OLIVES` or the config key `simulation.protein.olives` to the script. `prepare` stops with an error if no script is found. To use an elastic network instead, set `olives: null` and add `elastic: true` to the martinize options.

## Workflow

### 1. Prepare the system

```bash
martini-abfe prepare \
    --complex complex.pdb --ligand-resname LIG \
    --itp ligand.itp --gro ligand.gro \
    [--mapping mapping.ndx] [--config my-config.yaml] \
    -o systems/my-system
```

This creates the system directory and performs these steps:

1. Splits the complex into `protein.pdb` and `ligand-aa.pdb`. If the ligand residue occurs several times, select one copy with `--ligand-chain` and `--ligand-resid`.
2. Copies the ligand topology to `ligand.itp` and places the CG ligand at the crystal pose in `ligand.gro`. How it is placed depends on `--orient`:
   - `mapping` (default with `--mapping`): superimposes the beads onto the centers of their mapped atoms. The indices are 1-based and refer to the ligand atoms in the order they appear in the complex. Indices beyond the structure (e.g. missing hydrogens) are ignored.
   - `auto` (default without a mapping): fits the orientation of the CG ligand to the shape of the atomistic ligand.
   - `none`: only moves the CG centroid to the atomistic center of mass.
3. Coarse-grains the protein with martinize2 and OLIVES. All atoms other than `ATOM` records (waters, ions, cofactors) are removed. Only the first model and alternate location are kept. Multiple chains are merged into a single molecule; `--chain A,B` keeps only some of them.
4. Adds the pocket reference bead `DUM` at the ligand center of mass. It is a virtual site of the 8 backbone beads around the pocket that are spread farthest apart. If fewer than 8 beads lie within 1.1 nm, the cutoff is increased automatically.
5. Writes the full configuration to `config.yaml` in the system directory.

Check `protein-cg.pdb` and `ligand.gro` before running, e.g. in VMD. The summary printed by `prepare` gives the distance between the reference bead and the ligand center. It should be close to zero.

### 2. Run the simulations

```bash
martini-abfe run systems/my-system [systems/other-system ...]
```

Each system is simulated in two legs:

- **`protein`** (3 repeats): the ligand is decoupled in the pocket and kept there by a flat-bottom restraint (0.4 nm) to the reference bead.
- **`water`** (2 repeats): the ligand is decoupled in 0.15 M NaCl.

COBY places the solute at the center of a cubic box, with an edge of the solute diameter plus `box_margin`. The box is filled with water at its equilibrium density and with 0.15 M NaCl plus counter-ions for the net charge (`neutralize: false` adds none).

Charged ligands use the schedules with separate Coulomb decoupling. Minimization and equilibration run locally. The lambda windows are appended to a file queue, and `run` waits until they are finished. Start one worker per machine to process the queue:

```bash
martini-abfe-queue -f simulation-queue.dat --replex 1000 --n-gpus 2 --extra "-ntomp 8"
```

Without `--replex`, the windows run one by one, without replica exchange. To run everything locally without a queue, use `martini-abfe run --queue none ...`.

To only set up the legs and queue their windows, e.g. to process them later in a cluster job, use `martini-abfe run --no-wait ...`. It returns once all windows are queued; compute the free energies with `martini-abfe analyze` when the workers are done.

`run` logs what it is doing for each system and leg: box setup, minimization and equilibration (with their log files), the number of queued lambda windows and how many of them are finished. Use `martini-abfe -v run ...` for debug output.

Runs can be resumed: finished steps are skipped when `run` is called again. To rebuild a leg, e.g. after changing the force field paths, delete its directory (`protein-r1/`).

### 3. Analyze

`run` computes the free energies at the end. For systems run elsewhere, compute them with:

```bash
martini-abfe analyze systems/*/
```

The results are stored in each system directory:

- `info.json`: decoupling free energy of each leg and repeat, including the restraint and standard state corrections of the bound leg.
- `binding.json`: the binding free energy, ΔG_bind = ΔG_water − ΔG_protein, in kcal/mol. Its error is the standard error over repeats. It also contains the corrected binding free energy `dG_bind_corrected` and the details of the correction under `underbinding`.

### Underbinding correction

Martini 3 underbinds systematically, by 2.7 to 8.7 kcal/mol depending on the target ([arXiv:2610.11714](https://arxiv.org/abs/2610.11714)). The offset grows with the number of charged residues lining the pocket, N_charged:

ΔG_CG − ΔG_ref = 0.507 N_charged + 3.265 kcal/mol

The model was fitted to the mean offsets of 15 targets against FEP+ (R² = 0.78, leave-one-out Q² = 0.72). `dG_bind_corrected` subtracts this offset from ΔG_bind. Its error adds 0.83 kcal/mol in quadrature, the leave-one-target-out error of the predicted offset. Counting the pocket residues follows the paper:

- The pocket contains all residues with a heavy atom within 0.6 nm of any heavy atom of the crystal ligand (`protein-clean.pdb`, `ligand-aa.pdb`).
- Residue charges come from the protonation states of the atomistic structure. With hydrogens present, they are read from the hydrogens; otherwise, residue names (e.g. `ASH`, `HIP`) or standard states (charged Asp, Glu, Lys and Arg, neutral His) are used. Uncapped chain termini count as charged, so a residue at a terminus can change its net charge.

`prepare` prints the charged pocket residues and the predicted offset. It is the same for all ligands bound to the same pocket, so it improves absolute binding free energies but does not change the ranking of ligands within a target. The model was fitted to pockets with 0 to 10 charged residues, so treat predictions outside this range with care. Protonate the complex before `prepare`, so that the count reflects the intended protonation states.

## Configuration

`martini-abfe show-config` prints all settings. Pass a YAML file with the settings to change to `prepare --config` (see [examples/config.yaml](examples/config.yaml)). Common changes are:

- `gromacs`: mdrun arguments for minimization and equilibration, e.g. `-ntomp 8 -noddcheck -rdd 2.0`
- `martini`: force field paths
- `protein.box_margin`, `water.box_margin`, `*.salt`, `*.neutralize`: box size, NaCl concentration and counter-ions
- `protein.martinize`: any martinize2 option, e.g. `mutate: "HIS320:HIS"`
- `protein.n_reruns`, `water.n_reruns`: number of repeats
- `protein.dummy_bead`: construction of the reference bead
- `*.production`, `*.production_charged`: mdp files, for other lambda schedules

The default mdp files are in [src/martini_abfe/mdps](src/martini_abfe/mdps).

## System directory

| File                                  | Content                                                     |
| ------------------------------------- | ----------------------------------------------------------- |
| `config.yaml`                         | Complete configuration of the system                        |
| `complex.pdb`, `protein.pdb`, `ligand-aa.pdb` | Input complex and its split parts                   |
| `ligand.itp`, `ligand.gro`            | CG ligand, placed in the pocket                             |
| `protein-cg.pdb`, `protein.itp`       | CG protein with the reference bead                          |
| `protein_info.json`                   | Protein charge, ligand center, reference bead definition    |
| `info.json`, `binding.json`           | Ligand info and free energies, with the underbinding correction |
| `protein-r*/`, `water-r*/`            | Simulations of every leg and repeat, one `lambda-*` folder per window |

## Limitations

- Cofactors, metal ions and crystal waters are removed from the protein. Systems that need them in the pocket are not supported.
- Only one ligand can be bound. Other copies of the ligand residue are removed with the remaining non-protein atoms.
- The reference bead requires the ligand to be enclosed by protein backbone. `prepare` warns if the virtual site cannot reach the ligand center.
- The lambda schedules were optimized on the drug-like ligands of the OpenFE benchmark set. For very different ligands, check the overlap between neighboring windows.
