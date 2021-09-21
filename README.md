# WCGAN

The models changed some place based on the DCGAN

motivates:
1. remove the final activate functions on the Discriminator models
2. remove the log transforms on loss functions
3. clip the parameter of the Discriminator in -0.01 to 0.01
4. used the RMSprop optimizer
