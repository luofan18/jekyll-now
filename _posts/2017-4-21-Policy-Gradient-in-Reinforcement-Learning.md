---
layout: post
title: Policy Gradient in Reinforcement Learning
published: true
---

This article is a simple introduction for coding policy gradient algorithm and assumes you already have some knowledge about reinforcement learning and machine learning.

### Direct Parameterize Policy

There are several branches of methods in reinforcement learning. Apart from Q-learning, where you approximate the Q-function of the state and action, you can direct parameterize the policy. 

Given a vector of parameters \(\vec{\theta}\) on policy, you have a policy \(\pi(a \vert s,\vec{\theta})\) which give you the probability of certain action \(a\) under state \(s\). Then you sample an action from that distribution, take this action, observe the next state and reward. After you run the same policy for a period, you evaluate your current policy with data collected and figure out which actions are responsible for better reward. Then you increase the probability of 'good' policy and decrease the 'bad' policy.

### Parameterize the Probability of Actions under Given states

Obviously, the input to the policy $\pi(a|s,\vec{\theta})$ is the state we are currently in. Instead of using a policy that gives us a deterministic action, here we make the policy output a distribution on action. The reason is that by making the policy random we are actually exploring the world. Just like in Q-learning we use epsilon-greedy algorithm.

If the action space is discrete, where you have a fixed number of actions, the output of the $\pi(a|s,\vec{\theta})$ can be log probability of different action, like the output of neural network in classification problem. If the action space is continuous space, one way is to assume you action follows the normal distribution, where the mean of the normal distribution is calculted by your function. The variance of the normal distribution can be fixed or parameterized, too.

So, if you use a linear function to parameterize the policy, your code looks like this,
```python
# discrete action
a_logit = state * W
# continuous action
a_mean = state * W
```
Of course, neural networks are good choices to fit the policy, too.

### Measure 'Goodness' of Policy

As we hope that the 'good' actions happen more likely and 'bad' actions happen less likely we need to define a baseline for whether the action is good or not. A near optimal baseline is the expectation of discounted return we get just now using previous policy. This is exactly the value of the policy which we used to collect the data. Below is a simple implementation to calculate the discounted return along a path,
```python
# discount factor
gamma = 0.99
for path in paths:
	# calculate discounted return along a path
	path_len = len(path['reward'])
	reward = path['reward']
	# our discounted return
	return_t = []
	# do backward
	# first just add the discounted return in reverse order
	# the discounted return of last step
	return_t.append(reward[-1])
	# remaining ones
	for i in range(1, path_len):
		return_t.append(return_t[i-1] * gamma + reward[path_len-i])
	# reverse the return
	return_t = return_t[:,:,-1]
	path['return_t'] = return_t
```

Because we can only experience finite number of states or small number of states if the state space is very large, we can estimate the baseline by fit another linear function or neural networks.

Next, we can use the baseline to evaluate our current policy. First, run the current policy in the environment to collect data. Then calculate the discounted return for this path, and use baseline function to predict the value of each state in this path. Use the actual discount return to to subtract the predicted return and we know how good an action is compared to our old policy. The values we get is called advantage values. Code is
```python
import numpy as np
# for every path
for path in paths:
	adv_t = path['return_t'] - value_function.predict(np.vstack(path['state']))
	path['adv_t'] = adv_t

# it is good to normalize the advantage value
# concatenate all the advantage value to a numpy array
adv_t_all = np.hstack([path['adv_t'] for path in paths])
# normalize advantage value. 1e-8 for stabilizing the division
adv_t_all = (adv_t_all - adv_t_all.mean()) / (adv_t_all.std() + 1e-8)
```

And then we use the advantage values to help update the policy. We also need to use the new data to retrain the baseline function and use it as a baseline for the updated policy.

### Define Loss

Now, we have the advantage value to measures the actions. The loss for each action is the product between log-probability of actions taken and the advantage value.
```python
surr_loss = - a_logprob * adv_t_all
```    
Then we can compute the gradient of our policy by minimizing this loss.

If you wonder why we use the log-probability of action, I will derive it as last.

### Combine them all

Now you have everything you need for updating your policy. In all, the step to run the policy gradient method in an environment is

 1. Define an initial policy and initial value function
 2. Run the polity in some environment to collect some data. The data contain every state you experienced, the respect reward. Each episode is called a path.
 3. Use your value function and discounted return to calculate the advantage value
 4. Calculate loss and update your policy
 5. Fit your value function to new paths
 6. Repeat 2-5 until your policy is good enough

### Advanced Policy Update - Constrain Policy Update with KL Divergence

Because we use the value function trained on the old policy to predict the value of the new policy, we are introducing some bias, to ensure that our policy does not change too much. We can constrain the step size of policy update. When we find we made a big change to the policy, the decrease the step size. If the change is small, we increase the step size. A measurement of this change is KL divergence between our old policy and our updated policy. The formula to compute the KL divergence between two probability distribution is

$$ KL(p_1(x), p_2(x)) = \int{p_1(x)\log p_1(x)} - \int{p_1(x)\log p_2(x)}$$

For discrete action, the KL divergence is calculated as
```python
kl = a_prob_old * (a_logprob_old / a_logprob)
# take the mean of KL of all action
Kl = kl.mean()
```
For action sampled from the normal distribution, the solution is a little bit complicated. After some derivation, we code like this
```python
kl = np.log(a_std / a_std_old) + (a_std_old ** 2 + (a_mean_old - a_mean) ** 2) / (2 * a_std) - 0.5
``` 

Now, we can adaptively change the step size to update our policy. 
```python
# adaptively change step size of update, e.g. learning rate
if kl > desired_kl * 2: 
    stepsize /= 1.5
    print('stepsize -> %s'%stepsize)
elif kl < desired_kl / 2: 
    stepsize *= 1.5
    print('stepsize -> %s'%stepsize)
else:
    print('stepsize OK')
```

### Derivation of Loss 

Suppose you have a path $\tau$, the probability of which occurs under a certain policy is $\pi({\tau|\theta})$, the discounted return for this path is $R(\tau)$. The expected return of this policy is $E_{\tau \sim \pi(\theta)}[R(\tau)])$. Now, we want to improve this policy, e.g. improve the expectation. Let take the gradient of the expectation, respect to $\theta$, 

$$
\begin{aligned}
\bigtriangledown _\theta{E_{\tau \sim \pi(\theta)}[R(\tau)]} & = \bigtriangledown _\theta \int{\pi(\tau, \theta)R(\tau)d\tau}  \\
& = \int{\bigtriangledown _\theta \pi(\tau, \theta)R(\tau)} d\tau \\
& = \int{\pi(\tau, \theta) \frac{\bigtriangledown _\theta \pi (\tau, \theta)}{\pi (\tau, \theta)}R(\tau)} d\tau\\
& = \int{\pi (\tau, \theta) [\bigtriangledown _\theta \log( \pi( \tau, \theta)) R(\tau)}] d\tau \\
& = E_{\tau \sim \pi (\theta)}[\bigtriangledown \log( \pi (\tau, \theta))R( \tau)]
\end{aligned}
$$

The first and last step is exacly the definition of epectation. We do the 4th step because 

$$
\bigtriangledown _\theta \log (\pi (\tau, \theta)) = \frac {1}{\pi (\tau, \theta)} \bigtriangledown _\theta \pi (\tau, \theta)
$$

This fomula tell us that to adjust our policy, we can adjust the log-probability of the path times something. Subtract it with our baseline function $B( \tau) $, we have

$$
\begin{aligned}
\bigtriangledown _\theta E [R(\tau) - B(\tau)] & = E [ \bigtriangledown _\theta \log ( \pi ( \tau, \theta)) (R(\tau) - B(\tau))] \\
& = E [ \bigtriangledown _\theta \log( \pi( \tau, \theta))A(\tau)]  
\end{aligned}
$$

This is the loss we are actually using.
