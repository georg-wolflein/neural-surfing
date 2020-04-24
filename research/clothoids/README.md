Simple clothoid library
===

# Installation
```
pip3 install -r requirements.txt
```

# Usage
To use the demo, run
```
python3 draw_clothoids.py
```

To use the calculator in a Python script:
```python
from clothoidlib import ClothoidCalculator
calculator = ClothoidCalculator
params = calculator.lookup_points(a, b, c)
```