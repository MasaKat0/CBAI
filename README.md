# best-arm-identification

We rewrite the Julia code of Garivier and Kaufmann (2016) by using Python language.  
The code of Garivier and Kaufmann (2016) can be downloaded from
https://github.com/jsfunc/best-arm-identification.

## How to proceed ?

### mainBAI.py is the code for track-and-stop algorithm of Garivier and Kaufmann (2016).
main_contextualBAI.py the code for our proposed contextual track-and-stop algorithm.

### Choose the parameters of your bandit problem in mainBAI.py main_contextualBAI.py and before running these files. 
The code is mainly written for a bandit model with two contexts. You can rewrite the code for the case with more context easily.
#### zeta: probability of context.
ex. np.array([[0.5, 0.5]]).T
#### mu: marginalized mean rewards
ex. mu = np.array([[0.5, 0.45, 0.43, 0.4]]).T
#### mu_temp: conditional mean rewards
mu_temp = np.array([[0.5, 0.2, 0.2, 0.1]]).T  
The counter part is computed from zeta and mu.
#### delta: confidence.  
ex. delta = 0.1
#### num_similator: number of simulations.
ex. num_simulator = 20

# Configuration information

Experiments were run with the following python and scipy version: 

python: 3.7.5
scipy: 1.4.1


# MIT License of the Julia code of Garivier and Kaufmann (2016)

Copyright (c) 2016 [Aurélien Garivier and Emilie Kaufmann]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
