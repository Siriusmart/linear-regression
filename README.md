My implementaion of linear regression without following any tutorials. It works better or worse depending on the problem itself.

## How it works

Linear regression basically tries to find a straight line that fits all the cases, in this case there are multiple features (variables) that affect the output, the general formula would be.

Ŷ = θ<sub>1</sub>X<sub>1</sub> + θ<sub>2</sub>X<sub>2</sub> + θ<sub>3</sub>X<sub>3</sub> + θ<sub>4</sub>X<sub>4</sub> + ... + θ<sub>n</sub>X<sub>n</sub>

Where Ŷ is the predicted value (may not be accurate), X<sub>i</sub> are hte ith features (input variables), which their names doesn't matter, and θ<sub>i</sub> are the weight of the ith feature.

### Details

It changes the value of each of the weights individually, trying to get a lower error value. Basically "going down the slope" of the total error to weight graph. Note that we're not trying to get to 0, but just a lower number.

The weights are initialised randomly, and the process of changing the weights and checking are can be repeated any amount of times to achieve of more accurate results.

### Performance and accuracy
By lowering the learning rate (line 4), and increasing the iterations (line 58 and 62), I was able to achieve better results at cost of training time.

Running on an Intel i5 core single threaded, it took around 5 seconds and the average error is $183k plus or minus. Considering that this might not even be a suitable problem for linear regression, I'd consider this as "pretty darn good".
```
Accuracy after: 183624.60332932867
Results:

unit,date,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated|0.31412448468784077,5.24320157355406,0.42267329938880993,1.005468325652698,148.4647004207102,-0.7374299195459875,0.08251998521457053,0.7259587520934087,0.01793811926712599,0.2258190321395345,101.76150165971748,34.89614266198863,-31.42705749726021,0.05288131733113168
```

### Other features

By pasting in the `Results` line of the training output, it can be used to predict the price of house with given feature.

The feature of the item to be predicted should be separated by commas.

Finally, the data set is a modified version of [this](https://www.kaggle.com/datasets/shree1992/housedata) and everything I've learnt came from [this tutorial](https://youtu.be/NWONeJKn6kc) by freecodecamp.
