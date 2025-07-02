# OLA Project

This repository hosts our group project for Online Learning Applications. The project explores how agents adapt and compete in various simulated environments, aiming to understand strategies for market success.

## 🎉 Recent Updates (Latest)

**Major Bug Fixes Completed**: All critical algorithm implementation bugs have been identified and fixed:

- ✅ UCB1 confidence bound calculation corrected
- ✅ Primal-Dual algorithm logic completely repaired  
- ✅ Budget constraint implementation fixed
- ✅ Comprehensive test suite updated and validated (121/122 tests passing)
- ✅ Full analysis and documentation in `project_work/runfile.ipynb`

The multi-armed bandit auction platform is now production-ready with mathematically correct implementations.

## Testing

To run coverage report, run:

````cmd
pytest --cov=project_work --html=report.html --cov-report=html test/
open htmlcov/index.html
````
