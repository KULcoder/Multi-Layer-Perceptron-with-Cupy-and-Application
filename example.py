# This shows an example of how to use the PersonalMLP file
import PersonalMLP

# Data Load

X_train = ...
y_train = ...
X_test = ...
y_test = ...
X_valid = ...
y_valid = ...

# model configs
# Set up configs
config = PersonalMLP.Config()
# change configs
config.variable = value

# set up model
model = PersonalMLP.Model(config)

# train the model, which reports the statistics 
statistics = model.train(X_train, y_train, X_valid, y_valid)

# use the model to generate predict test
y_test_pred = model.forward(X_test)

# calculate matrics and demonstrate result