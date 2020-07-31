- ### Dice loss
```math
Dice_Coefficient = 2|A \cap B| / ( |A| + |B|) or 2|A \cap B| / ( |A| ^ 2 + |B| ^ 2) 
Dice_loss = 1 - Dice_Coefficient
Laplace smoothing: 1 - (2|A \cap B| + 1) / ( |A| + |B| + 1)
```
