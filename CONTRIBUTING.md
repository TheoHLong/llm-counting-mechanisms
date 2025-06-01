# Contributing to LLM Counting Mechanisms

We welcome contributions to this project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/llm-counting-mechanisms.git`
3. Create a virtual environment: `python -m venv env`
4. Activate it: `source env/bin/activate` (Linux/Mac) or `env\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`

## Development Setup

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where possible
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Testing
- Write tests for new functionality
- Ensure existing tests pass
- Test with multiple model types when applicable

## Types of Contributions

### 🐛 Bug Reports
- Use the issue template
- Include error messages and stack traces
- Provide minimal reproduction code
- Specify your environment (Python version, GPU, etc.)

### 💡 Feature Requests
- Check existing issues first
- Describe the use case and motivation
- Provide implementation suggestions if possible

### 🔧 Code Contributions
- Start with an issue discussion
- Create a feature branch: `git checkout -b feature/amazing-feature`
- Make atomic commits with clear messages
- Add tests and documentation
- Update README if needed

## Specific Areas for Contribution

### New Model Support
- Add new model evaluators in `src/model_benchmark.py`
- Test with representative examples
- Document any special requirements

### Analysis Methods
- Extend causal mediation analysis techniques
- Add new intervention strategies
- Improve activation patching methods

### Visualization
- Create new plot types
- Improve existing visualizations
- Add interactive plotting options

### Dataset Extensions
- Add new word categories
- Create specialized datasets
- Improve data generation methods

## Code Review Process

1. **Submit PR**: Create a pull request with a clear description
2. **Automated Checks**: Ensure CI passes
3. **Peer Review**: Wait for maintainer review
4. **Address Feedback**: Make requested changes
5. **Merge**: PR will be merged after approval

## Code Quality Standards

### Documentation
```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
        
    Returns:
        Description of return value
        
    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        True
    """
    pass
```

### Error Handling
- Use appropriate exception types
- Provide helpful error messages
- Log important events
- Clean up resources properly

### Performance
- Profile code for bottlenecks
- Use vectorized operations where possible
- Implement efficient data structures
- Consider memory usage for large models

## Testing Guidelines

### Unit Tests
```python
import unittest
from src.data_generation import CountingDataGenerator

class TestDataGeneration(unittest.TestCase):
    def setUp(self):
        self.generator = CountingDataGenerator()
    
    def test_example_generation(self):
        example = self.generator.generate_example("fruit", 5)
        self.assertEqual(len(example['list_items']), 5)
        self.assertIn('type', example)
```

### Integration Tests
- Test complete pipelines
- Verify model loading/unloading
- Check file I/O operations
- Validate visualization outputs

## Git Workflow

### Commit Messages
```
feat: add support for Gemma models
fix: resolve GPU memory leak in causal analysis
docs: update installation instructions
test: add unit tests for data generation
refactor: simplify model evaluation interface
```

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test additions

## Release Process

1. Update version numbers
2. Update CHANGELOG.md
3. Create release notes
4. Tag the release
5. Update documentation

## Community Guidelines

### Be Respectful
- Use inclusive language
- Be patient with newcomers
- Provide constructive feedback
- Help others learn

### Stay Focused
- Keep discussions relevant
- Use appropriate channels
- Search before posting
- Provide context

## Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Email**: [maintainer-email] for private concerns

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

Thank you for contributing to LLM Counting Mechanisms! 🚀
