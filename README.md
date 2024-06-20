# deQrypt

### An indicator based stock trend predictor built from scratch given the selected ticker.


An old hackathon of me and friends that was never finished, so I decided to remix it in my freetime to learn some web development.
To make predictions, trained a neural network based on data off of the past year to make predictions that are "relevant" to the current market trends.
I decided to cut out the initial proposal to include sentiments, as all the free financial news APIs only limit about calls that range about 2 months ago. Could be done later on still though by storing all the relevant data and peel them off one by one, but the lack of reference of time with the API that I found does not allow me to perform such a task.
The neural network was able to achieve about 60% given the data from 5 years ago until now, which was not too different from the built-in classifiers available.


Further Explorations:
- Investigate on neural network architecture with parameters, learning rates, batch size, etc
- Investigate on dataset on possible imbalance within dataset, or just completely overestimating the relevance of the indicators to stock trends (which can be true as real prop traders watch price action more than indicators as they are ***lagging***)
- Generate more graphs and provide on more information on why it might trend that way? Probably can still integrate recent news and explainations from LLMs, but not going to be too objectively (in my opinion) informative to be honest.
