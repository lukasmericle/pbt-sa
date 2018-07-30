# pbt-sa
Implementation of the Google DeepMind paper introducing population-based training, except applied to simulated annealing instead of neural networks. This implementation is set up to solve the multidimensional knapsack problem.

To run, simply execute `python ./sim/main.py` from the root directory. Upon first run, a `cfg.json` file will be generated and saved in the root directory. Use this to adjust the settings for running PBT.

Description of settings to come, if public demand exists.

## Preliminary results

The test run is compared against a baseline run of non-communicating processes. Both runs use initial hyperparameter settings drawn from identical distributions. 

The test run is shown in blue, and the baseline is shown in orange. The faint lines in the background are each individual worker process. The thick blue line is the average of the top 10% of workers, ranked by current value. The thick orange line is the #1 best worker at any given time, along with the hyperparameters associated with that best worker. The vertical dashed lines indicate the first time where the maximum value was attained in any worker process. The horizontal dashed lines indicate the maximum value.

![Preliminary results](https://github.com/lukasmericle/pbt-sa/blob/master/20180730030740%2B20180730031812-short.png)

The run using PBT achieves the maximum recognized value in less than one minute on my eight-year-old Lenovo Thinkpad T410s. The baseline run does not achieve the maximum recognized value in 10 minutes of running, and only attains its highest value after more than six minutes.

## Lessons learned

* Less is more. Trying to make the energy function complicated without focusing on other aspects of the algorithm will lead you astray of an elegant and functional solution.
* Defining the neighbor function for simulated annealing appropriately makes a huge difference. When possible, try to avoid selecting low-worth candidates by adjusting the neighbor function to generate only candidates which are "interesting" for the final solution. In this case, we add some random items to the current solution and remove random items until the knapsack constraints are satisfied. This greatly restricts the search space by considering only solutions where the knapsacks are near capacity.
