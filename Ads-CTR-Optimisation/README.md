# Ads CTR Optimisation using Reinforcement Learning

The Dataset contains information about ten differnt types of ad with differnt types of designs for the same product or service displayed to differnt users online once they connect to a certain website 
or to a search engine . Each Row corresponds to a differnt user on the internet. Since there are 10000 rows, we have data of ten thousand users.
0-> User clciked on the ad
1-> User din't click on the ad

In reality, what happens is that users connect one by one to the webpage and for each of them, we successively show them the ad. So everything happens in real time.
It's a dynamic real time process. It's not a static process with a static dataset which was recorded over a certain period of time.

This dataset is a simulation in the sense that each time a user connects to the webpage, it tells us even if we wouldn't know in reality on which ad the user would click on and this is the only way we can actually run the UCB algorithm or the Thomson Sampling algorithm 
to figure out the ad that has the highest conversion rate.The users here are represented as rounds so we need to figure out in a minimum number of rounds, which ad converts to the most meaning, which is the best ad to which the users are most attracted to.
And that's why we need a stronger algorithm than a simple statistics algorithm.

## Upper Confidence Bound Algorithm
 
 ### Step 1
 At each round n, we consider two numbers for each ad i:
 
 N<sub>i</sub>(n) - the number of times the ad i was selected up to round n
 
 R<sub>i</sub>(n) - the sum of rewards of the ad i up to round n
    
 ### Step 2 
 From these two numbers we compute:
 
   The average reward of ad i upto round n 
   
   r_avg<sub>i</sub>(n) = R<sub>i</sub>(n) / N<sub>i</sub>(n)
   
   The confidence interval [ r_avg<sub>i</sub>(n) - &Delta;<sub>i</sub>(n) , r_avg<sub>i</sub>(n) + &Delta;<sub>i</sub>(n) ] at 
   round n with 
   
   &Delta;<sub>i</sub>(n) = &radic; ( 3log(n) / 2 N<sub>i</sub>(n) )
         
 ### Step 3
 We select the ad i that has the maximum UCB r_avg<sub>i</sub>(n) + &Delta;<sub>i</sub>(n)

 ## Thompson Sampling Algorithm

### Step 1
 At each round n, we consider two numbers for each ad i:
 
 N<sub>i</sub><sup>1</sup>(n) - the number of times the ad i got reward 1 up to round n
 
 N<sub>i</sub><sup>0</sup>(n) - the number of times the ad i got reward 1 up to round n
 
 ### Step 2
 For each ad i, we take a random draw from the distribution below:
 
 &theta;<sub>i</sub>(n) = &beta;( N<sub>i</sub><sup>1</sup>(n) + 1 , N<sub>i</sub><sup>0</sup>(n) + 1 )
 
 ### Step 3 
 We select the ad that has the highest &theta;<sub>i</sub>(n)

