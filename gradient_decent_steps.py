import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

#sns.set_style("darkgrid")
#%matplotlib inline

def f(x):
    return x**2

def f_prime(x):
    return 2 * x

# gradient descent steps
starting_x=13.87
n_gradient_descent_steps = 100
learning_rate=0.02

steps = [starting_x]
for step in range(n_gradient_descent_steps):
  current_step = steps[-1]
  next_step = current_step - learning_rate * f_prime(current_step)
  steps.append(next_step)

steps = np.array(steps)
last_step = round(steps[-1], ndigits=2)
df_gradient_descent = pd.DataFrame({"y": f(steps)})

# plots
# fig, axes = plt.subplots(1, 2, figsize=(20, 6))
# plot_1, plot_2 = axes
# df.plot(legend=False, ax=plot_1)
# plot_1.plot(steps, f(steps), linestyle='--', marker='o', color='k')
# plot_1.set_xlabel("x")
# plot_1.set_ylabel("f(x)")
# plot_1.set_title("f(x) = x²", fontsize=16)
# plot_1.annotate("learning rate = {}".format(learning_rate), (-2, 90), fontsize=14)
# plot_1.annotate(f"Value of x at the last step: {last_step}", (-4, 85), fontsize=14)
# 
# df_gradient_descent.plot(legend=False, ax=plot_2)
# plot_2.set_ylabel("f(x)")
# plot_2.set_xlabel("Gradient Descent Step")
# plot_2.set_title("f(x) based on Gradient Descent Steps", fontsize=16)
# plot_2.annotate("learning rate = {}".format(learning_rate), (len(steps) / 2, f(steps[1])), fontsize=14)
print(steps)