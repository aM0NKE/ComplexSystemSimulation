# Group 8: “Schelling’s Segregationists”                              
(Just to be clear, we aren’t advocates of racial segregation.)

## Background of our project

Segregation between different population groups is modelled by Schelling’s segregation model using an agent’s preference for how many similar neighbours they would like to live next to. The original model focused on two population groups (majority/minority population) and showed that segregation happens when agents want 30% of neighbours to be similar to them. In our model, we would like to add a socio-economic element. We start out with a number of different population groups and give them some level of “wealth” normally distributed over all agents (we are considering the wealth distribution of the US as initial values). Agents still move based on the number of neighbours who look like them, meaning that when an agent has too few similar neighbours around them they will move to a random other location on the grid. The wealth of each agent will be influenced by the wealth of its neighbours. When an agent moves to a new location its new wealth will be determined by averaging its own wealth with the average wealth of its neighbours. We might tweak the economic rules still, a change we are discussing is that the wealth of an agent will only be altered when he is happy (and thus stays put) on his/her location. We will be looking at Moore neighbourhoods. This way we would like to model Socio-Economic Status, where an agent’s neighbours are their social network that influences that agent’s economic position. Importantly, the plan now is to see how the segregation as created by the Schelling model (with more than two population groups) corresponds with the wealth distribution. Will the wealth distribution differ between different population groups and how much? Will this difference be bigger when the segregation is higher? Does the specific distribution of these population groups influence this?

## Research questions
- Main RQ: Will segregation influence the wealth distribution between different population groups?

- Sub RQ1: *At what **tolerance threshold** does segregation occur?*
- Sub RQ2: *At what **population density** does segregation occur?*
- Sub RQ3: *At what **alpha** does wealth clustering occur*?
- Sub RQ4: *Is wealth clustering correlated to segregation clustering?*
- Sub RQ5: *What is the **critical population density** where percolation happens?* 
- Sub RQ6: *What is the **critical tolerance threshold** where percolation happens?*
- Sub RQ7: *What is the effect of alpha and homophily on the halftime?*
- Sub RQ8: *What is the effect of alpha and agent density on the halftime?*

## Hypotheses

- Main hypothesis: Segregation and the distribution of wealth are linked, thus segregation will change this distribution. 

## Which model

Schelling’s model of segregation (ABM): We modify the Schelling model of segregation to include multiple populations (between 1 and 10) and add a wealth distribution. We start with random/log-normal distributed wealth over the agents, independent of which population group they belong to and change this wealth every time an agent changes location and this change will be dependent on the wealth of its new neighbours. We think that our modified version of Schelling’s segregation model will exhibit some kinds of complex behaviour. 

## Emergent phenomenon of focus

 The emergent phenomenon we focused on is the difference in wealth distribution between the population groups when segregation is high. Our focus will mainly lie in investigating cluster sizes, segregation coefficient, Maran’s I statistic, wealth segregation and percolation to measure segregation and wealth inequality. We will also look at breaking symmetry in the sense of breaking the segregation/unequal wealth distributions.

## References

Schelling's original paper describing the model:

[Schelling, Thomas C. Dynamic Models of Segregation. Journal of Mathematical Sociology. 1971, Vol. 1, pp 143-186.](https://www.stat.berkeley.edu/~aldous/157/Papers/Schelling_Seg_Models.pdf)

An interactive, browser-based explanation and implementation:

[Parable of the Polygons](http://ncase.me/polygons/), by Vi Hart and Nicky Case.