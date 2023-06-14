# Exploration and Investigation

This AI recognizes the traffic images.
<br>
Run `python traffic.py gtsrb-small`
<br>
It can read a larger image file but the git does not accept it, so I pushed a smaller file.

| Experiment | Conv2D Config| Loss improvement | Accuracy improvement | Result                                                                           | Fix                                              |
| ------- | ----------------------- | ----------------- | ---------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------- |
| 1    | 16 filters, 3x3 kernel  | 3.5729 to  0.2678    | 0.1754 to  0.9154         | Pretty good job                                                           | Applied more filters.               |
| 2     | 32 filters, 3x3 kernel  | 3.5673 to  0.2328       | 0.2406 to  0.9327      | Reached the previous best accuracy 3 epochs earlier, start with better accuracy                                                             | Applied more filters.                              |
| <span style="color:red">3    | 64 filters, 3x3 kernel | 2.4374 to  0.2430     | 0.5099 to 0.9366       | Loss number is less than 3 for the 1st time, start with better accuracy | Change dropout to 0.7 | 
| 4     | 64 filters, 3x3 kernel | 4.0110 to 1.1987   |  0.1573 to 0.6267     | the worse start and result, learning time is slow  | Set back dropout to 0.5, set hidden layers unit 128 to 8, increase number of filters|
| 5     | 128 filters, 3x3 kernel |  4.0491 to 3.4975|    0.0541  to 0.0556    | Model does not learn although the leaning time is super slow | change output activation from softmax to sigmoid, decrease filter number  |
| 6     | 64 filters, 3x3 kernel |   2.8018 to 0.3001|    0.4800  to 0.9149    | the 2nd best result following experiment 3 |  set back sigmoid to softmax  |


## The structure of my model
- Conv2D, ReLU activation
- MaxPooling with 2 x 2 kernel/ filter
- Hidden layer
- Dropout
- Output layer (activations: sigmoid, softmax) 


## Overall
<br>
64 filters for both the 1st and 2nd layers with 3x3 kernel, hidden layer with 128 units, the 0.5 dropout rate brought the best result. 
I thought the more filters the better but it does not always apply that way.
The balance is more important, and dropout rate and more output layer units help accuracy.
This is very interesting practice for image learning with machine. Next I'd like to fathom various actvation methods. 
 
