Thank you very much for your quick reply,

I may not have been very clear in the first point : I try to study the resistance (robustness?) of a given image in the dataset to an adversarial attack, rather that the resistance of the whole classifier.

Here is a better explaination of what I "found":


I trained two models to test my results, so they are basic, with a simple AlexNet architecture. The model trained on MNIST makes 62 errors on the 10k test images, and the model trained on FashionMNIST makes 876 errors.


The attack consists in changing the models prediction, without knowing the true label of course. Assuming that Conf is the confidence of the network on the predicted label for a given image, the perturbation is made by 500 steps of gradient descent on a perturbtaion r with the following loss function:

- if Conf < 0.2 then loss = |r|
- else if Conf < 0.9 then loss = Conf + |r|
- else loss = Conf - |r|

(the third case, loss = Conf - |r| avoids the perturbation to stall at 0)


The perturbation r is initialized to 0, so as not to influence in any way the direction preferred by the gradient descent.


I first observed that some attacks are much harder than others to carry out. Here (https://imgur.com/GaSFOe3), the image on the left may be qualified as "hard" to attack, while the image on the right is "easy" to attack. (in blue the conf of the network on the initial image prediction, and in red the norm of the perturbation through the attack.)

To quantify more mathematically the "difficulty to attack" a given image (that I called the resistance, but this term might be prone to confusion), I introduce three possible expressions of such resistance:

- Res_N is the value of |r| at the end of the attack,
- Res_max is the max value of |r| during the attack,
- Res_min is the minimal steps required to change the prediction of the network

The first interesting result that I observed is the very strong correlation between the resistance to an attack and the corectness of the network prediction (on an image, not the overall accuracy):

On MNIST, on the classifier I trained, I found that the value of the resistance discriminates quite well which image are correctly classified, and which are not: for instance, 90% of the correctly classified images have a value of res_N greater than 0.97, and 90% of the incorrectly classified images have a value of res_N lesser than 0.57.

This result might be an interesting way to have a more reliable value for the confidence of the network: adversarial examples themselves show that an error can give a wrong prediction with a 99% accuracy. On the other hand, the value of the resistance seems to give a much better estimation of whether the network fails or not. Do you think that this idea might be relevant?


The second result, that I called "adversarial correction", is the fact that when an image that is incorrectly classified is attacked, the predicted category of the adversarial example obtained is almost always the real label of the initial image.

Thus, with previous networks: out of the 62 errors committed on the MNIST test database, 53 are corrected by the adversarial corrections; and out of the 876 committed on FashionMNIST, 613 are corrected, i.e. 85% and 70% respectively.

This instantly converts the Top 1 errors of respectively 0.53% and 8.7% for MNIST and FashionMNIST to Top 2 errors of 0.09% and 2.6% respectively.

This phenomenon has an intuitive explaination : as the attacks consists in changing the prediction of the network, and because the perturbation is initialized at 0, favoring no direction, the "easiest" path to change the prediction of the network is to correct its error!

I think that this might be a good strategy in a problem such as the ImageNet challenge (because of its Top 5 error metric): I could "adversarially correct" the best three prediction of the network for each image to get three other predictions, and submit five of those six obtained. I don't have enough computational power right now, but I think that this approcah could lead to better results given any classifier.


I then tried to combine these two ideas to directly improve the Top 1 error of any classifier: for each image, determine its resistance, figure from it whether the image is correctly classified or not, and if not then return the adversarial correction of the image. This has not worked as well as I intended but I'm still on it!

Do you think that those results might be a good way to improve a given classifier? It also seems that the resistance can "detect" adversarial examples too, but I have not yet tested this assumption on a large enough dataset to be sure of my assumption.


Maximilien

---

> Hello,

> I am a 20 year old undergrad math student in France, and for a side project I decided to work on improving neural networks' robustness to adversarial examples.

> I think that I might have found some interesting results, and that's why I wanted to contact someone more articulate than myself in this area, to get an idea of the relevance of my findings. Do you think that you could give me some of your time?

> I made my experiments on both MNIST and FasionMNIST (didn't had enough resources to try on a more demanding dataset). After building basic classifiers for those datasets, I set up an adversarial attack algorithm (by gradient descent) that led me to two interesting results:

> - First, there  seems to be a strong correlation between the difficulty in carrying out this attack (that I quantified by the concept of resistance) and the accuracy of the network prediction. The "resistance" to an attack might be an interesting way to get a more comprehensive value of the confidence of a classifier network's prediction.


Do you see this when you compare two different trained models to each other, or when you compare one model early in training to itself later in training?

I've seen that fully trained models tend to have better adversarial robustness when they have better accuracy, but I've also seen that a model early in training has much better robustness than the same model late in training.
 


> - Second, the same adversarial attack, when carried out from an improperly classified image, almost always produces an image whose new category is that of the initial image. I call this phenomenon "adversarial corrections". On MNIST and FashionMNIST, adversarial corrections seem to rectify about 80% of the classification errors!


Is the attack based on maximizing the loss associated with the true label? Or based on changing the model's prediction?
Do you mean that the appearance of the input changes, or that the label assigned by the model changes?
Have you tried measuring the "top-2 accuracy" for the model (the fraction of examples where the correct class is among the two classes that the model assigns the highest probability) ?
 


> I would very much like to discuss this further with you, if you had the time, and would be very happy to have your criticism or advice on this subject.

> Thank you for your time,

> Maximilien

> P.S. Do you, by any chance, speak French? This would make communication a bit easier for me!


Juste un peu, c'est claire que tu parle l'anglais beaucoup meilleur que je parle le francais. Mais je peux essayer si tu veut