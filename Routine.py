import torch
from torch.optim import Adam

from TransNet import SARDINet, Discriminator
from utils import eval_loop, training_loop, visIm, checkpoint, adversarial_training__loop

from datetime import datetime

class TrainRoutine :

    def __init__(self, paths, hyperparameters, dataloaders, modelParams, optimization, device = "gpu"):

        self.init_params(paths, hyperparameters, dataloaders, modelParams, optimization)
        self.adversarial = modelParams["adversarial"]
        self.device = device

        #Model creation
        if len(self.modelPath)>0 :
            self.model = torch.load(self.modelParams["modelPath"])
        else :
            self.model = SARDINet()
        self.model.to(self.device)        #Transfer the model to the selected device
        self.optim = Adam(self.model.parameters(), lr = self.lr_main)

        #Discriminator (for adversarial training)
        if self.adversarial :
            self.discr = Discriminator()
            self.discr.to(self.device)
            self.optim_discr = Adam(self.discr.parameters(), lr = self.lr_discr)


    def init_params(self, paths, hyperparameters, dataloaders, modelParams, optimization):

        self.pathInput = paths["pathInput"]
        self.pathGT = paths["pathGT"]
        self.saveDir = paths["saveDir"]
        
        self.epochs = hyperparameters["epochs"]
        self.percentages = hyperparameters["percentages"]
        self.batch_size = hyperparameters["batch_size"]
        self.im_size = hyperparameters["im_size"]
        self.lr_main = hyperparameters["lr_main"]
        self.lr_discr = hyperparameters["lr_discr"]
        self.lambdaVal = hyperparameters["lambdaVal"]
        
        self.dataloader = dataloaders["dataloader"]
        self.dataloader_eval = dataloaders["dataloader_eval"]
        self.dataloader_test = dataloaders["dataloader_test"]
        
        self.modelPath = modelParams["modelPath"]

        self.loss_fn = optimization["loss_fn"]
        self.loss_discr = optimization["loss_discr"]


    def trainNet(self):
        if self.adversarial :
            self.adversarialTraining()
        else :
            self.classicalTraining()

    
    def classicalTraining(self):
        currDate = datetime.now()

        #Evaluation of the untrained network
        self.TrainLosses = [eval_loop(self.dataloader, self.model, self.loss_fn, self.device)]
        self.EvalLosses = [eval_loop(self.dataloader_eval, self.model, self.loss_fn, self.device)]


        #Starting training
        for e in range(self.epochs) :
            print("\nEpoch " + str(e+1) + "/" + str(self.epochs))

            #Move network to the selected device
            self.model.to(self.device)

            #Time measurement
            t1 = datetime.now()
            
            #One step of training
            TLoss = training_loop(self.dataloader, self.model, self.loss_fn, self.optim, self.batch_size, self.device)
            
            #Save metrics
            self.TrainLosses.append(TLoss)
            
            #Time measurement
            t2 = datetime.now()
            
            #Evaluate the training
            ELoss = eval_loop(self.dataloader_eval, self.model, self.loss_fn, self.device)
            
            #Save metrics
            self.EvalLosses.append(ELoss)
            
            #Time measurement
            t3 = datetime.now()

            print("Elapsed time Training : " + str(t2-t1) + " | Elapsed time Evaluation : " + str(t3-t2))

            #Save the model every 10 epochs
            if not (e+1) % 10 :
                checkpoint(self.model, self.TrainLosses, self.EvalLosses, e+1, self.saveDir)
                visIm(self.model, self.dataloader.dataset, e+1, self.saveDir+"Training/")
                visIm(self.model, self.dataloader_eval.dataset, e+1, self.saveDir+"Eval/")

        #Save the final model
        checkpoint(self.model, self.TrainLosses, self.EvalLosses, e, self.saveDir)
        visIm(self.model, self.dataloader.dataset, e+1, self.saveDir+"Training/")
        visIm(self.model, self.dataloader_eval.dataset, e+1, self.saveDir+"Eval/")

        #Time measurement
        tfin = datetime.now()

        print("Full Training Elapsed Time : " + str(tfin - currDate))


    
    def adversarialTraining(self):
        currDate = datetime.now()

        #Evaluation of the untrained network
        self.TrainLosses = [eval_loop(self.dataloader, self.model, self.loss_fn, self.device)]
        self.EvalLosses = [eval_loop(self.dataloader_eval, self.model, self.loss_fn, self.device)]

        self.TrainLosses[0] = (self.TrainLosses[0][0], self.TrainLosses[0][1], self.TrainLosses[0][2], 0)

        #Start adversarial training
        for e in range(self.epochs) :
            print("\nEpoch " + str(e+1) + "/" + str(self.epochs))

            #Move network to the selected device
            self.model.to(self.device)

            #Time measurement
            t1 = datetime.now()
            
            #Models, loss and optimizers used for training
            models = [self.model, self.discr]
            loss_fns = [self.loss_fn, self.loss_discr]
            optims = [self.optim, self.optim_discr]

            #Training step
            TLoss = adversarial_training__loop(self.dataloader, models, loss_fns, optims, self.batch_size, self.device, lambdaVal=self.lambdaVal)
            
            #Save the loss
            self.TrainLosses.append(TLoss)
            
            #Time measurement
            t2 = datetime.now()
            
            #Evaluation of the training
            ELoss = eval_loop(self.dataloader_eval, self.model, self.loss_fn, self.device)
            
            #Save metrics
            self.EvalLosses.append(ELoss)

            #Time measurement
            t3 = datetime.now()

            print("Elapsed time Training : " + str(t2-t1) + " | Elapsed time Evaluation : " + str(t3-t2))

            #Save the model every 10 epochs
            if not (e+1) % 10 :
                checkpoint(self.model, self.TrainLosses, self.EvalLosses, e+1, self.saveDir)
                visIm(self.model, self.dataloader.dataset, e+1, self.saveDir+"Training/")
                visIm(self.model, self.dataloader_eval.dataset, e+1, self.saveDir+"Eval/")

        #Save the final model
        checkpoint(self.model, self.TrainLosses, self.EvalLosses, e, self.saveDir)
        visIm(self.model, self.dataloader.dataset, e+1, self.saveDir+"Training/")
        visIm(self.model, self.dataloader_eval.dataset, e+1, self.saveDir+"Eval/")

        #Time measurement
        tfin = datetime.now()
        print("Full Training Elapsed Time : " + str(tfin - currDate))


    def inferNet(self):
        self.model.to(self.device)
        TestLoss = eval_loop(self.dataloader_test, self.model, self.loss_fn, self.device)
        visIm(self.model, self.dataloader_test.dataset, -1, self.saveDir+"Test/", nbIm = -1)