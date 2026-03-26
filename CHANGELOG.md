# Changelog

All notable changes to this project will be documented in this file.

## [1.3.3] - 2026-03-26

### Fixed
- **H5 file loading error**: Fixed `TypeError: Accessing a group is done with bytes or str, not slice`
  - The issue was incorrect slicing when accessing H5 file groups, caused by improper group path handling
  - Corrected the group accessing logic to properly traverse nested H5 file structure
  - Now loads all standard H5 format datasets correctly without exception

- **Resolution extraction error**: Fixed `IndexError: Tuple index out of range`
  - The issue caused incorrect resolution extraction (falsely detected as 4096) when parsing dataset metadata
  - Fixed dimension detection logic to handle different H5 metadata formats
  - Correctly extracts resolution from both single-file and double-file H5 datasets

## [1.3.2] - 2026-03-26

### Fixed
- **H5 file parsing bug**: Fixed incorrect data key parsing in H5 format for Zenodo datasets
  - The issue caused wrong tensor shape when reading Navier-Stokes data from Zenodo H5 files
  - Corrected key lookup to match the actual data format in the downloaded files
  - Now fully supports all H5 formats: single file (PDEBench) and double file (Zenodo)

## [1.3.1] - 2026-03-26

### Changed
- **Documentation fix**: Clarify "accuracy improvement" instead of "performance improvement" (we improve accuracy by reducing test loss, not speed)
- **Added English README**: `README_EN.md` for international audience
- **Cleanup**: Remove entire `internal/` directory for cleaner commercial release

### Project Structure after cleanup:
```
benchmark/
├── README.md
├── BENCHMARK_GUIDE.md
├── data_loader.py     # Universal data loader
├── run_benchmarks.py  # Benchmark entry point
├── generate_data.py
├── generate_ns_velocity.py
├── test_mhf_coda_pino_quick.py
└── data/             # Data directory
```

## [1.3.0] - 2026-03-26

### Added
- **Zenodo Double H5 Support**: Full support for downloaded datasets from https://zenodo.org/records/13355846
- **Universal Data Loader**: New `data_loader.py` supports all formats:
  - PT single file (local generated)
  - PT double files (train + test separate)
  - H5 single file (PDEBench format)
  - **H5 double files (Zenodo format) ✨** - train.h5 + test.h5 separate files
- **New command line arguments**: `--train_path` and `--test_path` for double file mode

### Changed
- Moved `run_benchmarks.py` from `internal/` to `benchmark/` root for direct access
- Updated documentation with clear usage instructions for Zenodo datasets
- Cleanup directory structure for commercial release

### Usage Examples

**Burgers 1D from Zenodo:**
```bash
python run_benchmarks.py \
    --dataset burgers --format h5 \
    --train_path ./data/1D_Burgers_Re1000_Train.h5 \
    --test_path ./data/1D_Burgers_Re1000_Test.h5
```

**Navier-Stokes 2D from Zenodo:**
```bash
python run_benchmarks.py \
    --dataset navier_stokes --format h5 \
    --train_path ./data/2D_NS_Re100_Train.h5 \
    --test_path ./data/2D_NS_Re100_Test.h5
```

## [1.2.0] - 2026-03-26

### Added
- **PINO Physics Constraints**: Complete Navier-Stokes equation residuals
  - Time derivative ∂u/∂t
  - Advection term (u·∇)u
  - Diffusion term ν∇²u
  - Incompressibility constraint ∇·u = 0
- **Real NS Data Support**: Velocity field time series [N, T, 2, H, W]
- **MHF+CoDA+PINO Integration**: 36% performance improvement on real NS data

### Changed
- Updated performance benchmarks with real NS data results
- Improved documentation with MHF vs MHF+CoDA comparison
- Enhanced examples with real NS data usage

### Performance
- **Darcy 2D**: +8.17% vs FNO, -48.6% parameters
- **Burgers 1D**: +32.12% vs FNO, -31.7% parameters
- **NS 2D (Real+PINO)**: +36% vs MHF+CoDA, -49% parameters ⭐

## [1.1.0] - 2026-03-24

### Added
- Cross-Head Attention (CoDA) mechanism
- Multi-head frequency decomposition (MHF)
- Support for NeuralOperator 2.0.0

### Performance
- Darcy 2D: +8.17% vs FNO
- Burgers 1D: +32.12% vs FNO
- NS 2D (Conservative): ~0% vs FNO

## [1.0.0] - 2026-03-20

### Added
- Initial MHF-FNO implementation
- Basic multi-head factorization
- Compatible with NeuralOperator framework

---

## Version History

| Version | Date | Key Features | Notes |
|---------|------|--------------|-------|
| **1.3.3** | **2026-03-26** | **H5 loading bug fixes** | Fix H5 group access and resolution extraction ✅ |
| **1.3.2** | **2026-03-26** | **H5 file parsing bug fix** | Fix incorrect data key lookup for Zenodo datasets ✅ |
| **1.3.1** | **2026-03-26** | **Documentation cleanup + English README** | Clean commercial release ✨ |
| **1.3.0** | **2026-03-26** | **Zenodo H5 double file support + Universal data loader** | Ready for external dataset validation |
| 1.2.0 | 2026-03-26 | PINO + Real NS Data | +36% accuracy improvement (NS+PINO) |
| 1.1.0 | 2026-03-24 | MHF+CoDA | +8-32% accuracy improvement |
| 1.0.0 | 2026-03-20 | MHF Foundation | Baseline |
