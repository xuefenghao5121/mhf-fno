# Changelog

All notable changes to this project will be documented in this file.

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

| Version | Date | Key Features | Performance |
|---------|------|--------------|-------------|
| 1.2.0 | 2026-03-26 | PINO + Real NS Data | +36% (NS+PINO) |
| 1.1.0 | 2026-03-24 | MHF+CoDA | +8-32% |
| 1.0.0 | 2026-03-20 | MHF Foundation | Baseline |
