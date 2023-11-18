const to_be_defined = null;

const schema = {
    "Machine Learning": {
        "Process": {
            "Train Model on data": to_be_defined,
            "Data Preperation": to_be_defined,
            "Analysis/Evaluation": to_be_defined,
            "Data Collection": to_be_defined,
            "Serve Model": to_be_defined,
            "Retrain Model": to_be_defined,
        },
        "Resources": {
            "Code": [
                "https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5",
                "https://missing.csail.mit.edu/",
                "https://teachyourselfcs.com/",
                "https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/section-3-structured-data-projects/end-to-end-heart-disease-classification.ipynb",
                "https://github.com/dformoso/sklearn-classification",
                "https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb",
                "https://www.fast.ai/",
                "https://course.fast.ai/",
                "https://www.deeplearning.ai/",
                "https://cs50.harvard.edu/ai/2020/",
                "https://fullstackdeeplearning.com/march2019",
                "https://www.youtube.com/watch?v=rfscVS0vtbw",
                "https://zerotomastery.io/",
                "https://www.pythonlikeyoumeanit.com/",
                "https://www.mrdbourke.com/get-your-computer-ready-for-machine-learning-using-anaconda-miniconda-and-conda/",
                "https://realpython.com/python-virtual-environments-a-primer/",
                "https://jupyter-notebook.readthedocs.io/en/stable/",
                "https://www.mrdbourke.com/mlcourse/",
                "https://www.kaggle.com/learn",
                "https://www.datacamp.com/",
                "https://www.dataquest.io/",
            ],
            "Books": [
                "Automate the Boring Stuff with Python, 2nd Edition: Practical Programming for Total Beginners 2nd Edition",
                "Python-Data-Analysis-Wrangling-IPython",
                "https://jakevdp.github.io/PythonDataScienceHandbook/",
                "Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems",
                "Deep Learning for Coders with fastai and PyTorch: AI Applications Without a PhD",
                "http://neuralnetworksanddeeplearning.com/index.html",
                "Introduction to Machine Learning with Python: A Guide for Data Scientists Paperback",
                "https://mlbook.explained.ai/",
                "https://www.buildingmlpipelines.com/",
                "https://christophm.github.io/interpretable-ml-book/intro.html",
                "https://seeing-theory.brown.edu/index.html#firstPage",
                "https://mml-book.github.io/",
                "https://explained.ai/matrix-calculus/index.html",
            ],
            "Concepts and process": [
                "https://www.elementsofai.com/",
                "https://developers.google.com/machine-learning/crash-course",
                "https://ai.google/build",
                "https://research.facebook.com/blog/2018/05/the-facebook-field-guide-to-machine-learning-video-series/",
                "https://madewithml.com/topics/",
                "https://workera.ai/",
                "https://www.kaggle.com/competitions",
                "https://karpathy.medium.com/software-2-0-a64152b37c35",
            ],
            "Mathematics": [
                "https://www.youtube.com/playlist?list=PLtmWHNX-gukIc92m1K0P6bIOnZb-mg0hY",
                "https://www.khanacademy.org/math/multivariable-calculus",
                "https://www.3blue1brown.com/topics/linear-algebra",
                "https://www.khanacademy.org/math/calculus-1",
                "https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices",
                "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi",
                "https://www.khanacademy.org/math/statistics-probability",
            ],
            "Rules and Tidbits": [
                "https://fastpages.fast.ai/",
                "https://pages.github.com/",
                "https://karpathy.github.io/2019/04/25/recipe/",
                "https://developers.google.com/machine-learning/guides/rules-of-ml/",
            ],
            "Datasets": [
                "https://datasetsearch.research.google.com/",
                "https://www.kaggle.com/datasets",
                "https://www.dataquest.io/blog/free-datasets-for-projects/",
                "https://storage.googleapis.com/openimages/web/index.html",
                "https://github.com/HumanSignal/awesome-data-labeling",
                "https://index.quantumstat.com/"
            ],
            "Papers": [
                "https://arxiv-sanity-lite.com/",
                "https://madewithml.com/",
                "https://paperswithcode.com/",
                "https://sotabench.com/",
            ],
            "Others": [
                "https://www.coursera.org/learn/learning-how-to-learn/",
                "https://www.mrdbourke.com/6-techniques-which-help-me-study-machine-learning-five-days-per-week/",
                "https://www.mrdbourke.com/aimastersdegree/",
            ],
            "Cloud Services": [
                "https://www.pluralsight.com/cloud-guru",
                "https://cloud.google.com/learn/training",
                "https://aws.amazon.com/training/",
                "https://learn.microsoft.com/en-us/training/azure/"
            ],
        },
        "Problems": {
            "Categories": {
                "Supervised": "labelled data",
                "Unsupervised": "finding patterns in unlabelled data",
                "Reinforcement Learning": "training using reinforcement",
                "Transfer Learning": "transferring knowledge",
            },
            "Regression": {
                "Example Problem": "stock price prediction, linear predictions",
                "Evaluation Metrics": {
                    "R-squared error": "perfect model scores 1.0",
                    "MSE (Mean Squared Error": "makes outliers stand out more, negative won't cancel positive, being 10% off is way higher than double of being 5% off",
                    "MAE (Mean Absolute Error)": "All errors on same scale, predicting 99 instead of 100 is same as predicting 101 instead of 100",
                    "RMSE (Rooted Mean Squared Error)": "same as MAE, rooted so gets in the same unit",
                }
            },
            "Dimensionality Reduction": "reducing number of features without loosing much meaning",
            "Clustering": "grouping peronas based on activity",
            "Sequence to Sequence (seq2seq)": "example nlp translation",
            "Classification": {
                "Problems": {
                    "Binary Classification": "cat or dog (true/false)",
                    "Multi-class Classification": "red, blue or green (multiple choises)",
                    "Multi-label Classification": "items in a photo (multiple, tags, keywords etc)",
                },
                "Evaluation Metrics": {
                    "Confusion Matrix": "useful for seeing where a model is getting confused",
                    "Accuracy": "out of 100 how many got it right, unreliable when class count is unbalanced, in that case use precision and recall instead",
                    "F1 Score": "The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0, f1_score = 2 * (precision * recall) / (precision + recall)",
                    "Precision": "accuracy of the positive predictions, a model with 1.0 precision score has no false positives",
                    "Recall": "ration of true positive predictions, a model with 1.0 recall has no false negatives",
                    "Precision Recall Tradeoff": "Increasing precision reduces recall, and vice versa, optimizing for one will reduce the other",
                    "Precision Recall Curve": "Plotting precision vs recall at different classification thresholds allow you to choose a precision value at a given recall value (as one increases, the other one decreases)",
                    "ROC Curve / AOC": "The receiver operating characteristic (ROC) curve is a common way to evaluate binary classifiers, Plots false positive rate against true positive rate, AUC (Area under the curve) 1.0 is perfect classifier, generally if don't have much positive examples or you care about false positive than false negative then use precision/recall curve rather than ROC/AUC curve",
                }
            },
        },
        "Tools": {
            "Toolbox": {
                "pretrained models": {
                    "Tensorflow Hub": "https://www.tensorflow.org/hub/",
                    "Pytorch Hub": "https://pytorch.org/hub/",
                    "HuggingFace Transformers (NLP)": "https://huggingface.co/docs/transformers/index",
                    "Detectron2 (CV)": "https://github.com/facebookresearch/detectron2"
                },
                "experiment tracking": {
                    "Tensorboard": "https://www.tensorflow.org/tensorboard/",
                    "Dashboard by weights and biases": "https://wandb.ai/site/experiment-tracking",
                    "neptune.ai": "https://neptune.ai/"
                },
                "Data & Model tracking": {
                    "Artifacts by weights and biases": "https://wandb.ai/site/artifacts",
                    "DVC": "https://dvc.org/"
                },
                "Cloud Services": {
                    "Google Colab": "https://colab.research.google.com/",
                    "AWS": "https://aws.amazon.com/machine-learning/",
                    "AWS Sagemaker": "https://aws.amazon.com/sagemaker/",
                    "GCP AI Platform": "https://cloud.google.com/vertex-ai",
                    "Azure ML": "https://azure.microsoft.com/en-us/products/machine-learning/"
                },
                "Hardware, building": {
                    "Choosing GPU": "https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/",
                    "Building a PC for DL": "https://medium.com/the-mission/how-to-build-the-perfect-deep-learning-computer-and-save-thousands-of-dollars-9ec3b2eb4ce2"
                },
                "Auto ML": {
                    "TPot": "https://github.com/EpistasisLab/tpot",
                    "Google Cloud AutoML": "https://cloud.google.com/automl/",
                    "Microsoft automated ML": "https://azure.microsoft.com/en-us/products/machine-learning/automatedml/",
                    "Sweeps by weights and biases": "https://wandb.ai/site/sweeps",
                    "Keras Tuner": "https://www.tensorflow.org/tutorials/keras/keras_tuner"
                },
                "Explainability (Why did my model do what it did)": {
                    "What-If tool": "https://pair-code.github.io/what-if-tool/",
                    "SHAP values": "https://github.com/shap/shap"
                },
                "Machine Learning Lifecycle": {
                    "Streamlit": "https://streamlit.io/",
                    "MLflow": "https://mlflow.org/",
                    "Kubeflow": "https://www.kubeflow.org/",
                    "Seldon": "https://www.seldon.io/",
                    "Production LevelDL Example Repo": "https://github.com/alirezadir/Production-Level-Deep-Learning"
                }
            },
            "Libraries": [
                "scikit-learn",
                "pytorch",
                "pytorch lightning",
                "tensorflow",
                "keras",
                "ONNX"
            ],
        },
        "Mathematics": {
            "Linear Algebra": "creating objects and set of rules to manipulate these objects",
            "Matrix Manipulation": "ml data is eventually turned into matrices and tensors",
            "Multivariate Calculus": "foundation for optimizing function like cost functions, with respect to multiple parameters",
            "Probability Distribution": "probabilities, sample space, series of possible events, random events",
            "Pobability": "study of uncertainity",
            "Optimization": "finding the ideal pattern which best describes the dataset, how do you optimize the model to do so",
            "Chain Rule": "basis of backpropagation, how neural networks improve themselves"
        },
    }
}
