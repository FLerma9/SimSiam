from torch import nn
from torchvision.models import vgg13, resnet34, alexnet, squeezenet1_1, mobilenet_v2

#Proyector MLP formada por 3 capas MLP.
class proyector_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=2048):
        super().__init__()
        hidden_dim = 2048
        #Primera capa
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        #Segunda capa
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        #Tercera capa
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

#Predictor MLP formado por 2 capas
class predictor_MLP(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        hidden_dim = int(in_dim / 4)
        #Primera capa
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        #Segunda capa
        self.layer2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

#Clase de la red siamesa SimSiam
class SimSiam(nn.Module):
    def __init__(self, arch='resnet34', dimension=2048):
        super(SimSiam, self).__init__()
        self.arch = arch
        #Definimos el back de SimSiam con la arquitectura que queremos
        self.encoder = SimSiam.get_backbone(arch)
        
        #Dependiendo de la arquitectura de la red la entrada del proyector MLP varía
        #Quitamos las capas clasificadoras del backbone y añadimos el proyector
        if 'resnet' in arch:
            out_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = proyector_MLP(out_dim, dimension)
        elif 'vgg' in arch:
            out_dim = self.encoder.classifier[0].weight.shape[1]
            self.encoder.classifier = proyector_MLP(out_dim, dimension)
        elif 'squeezenet' in arch:
            #Only if 224x224 input img 
            out_dim = 13*13*512
            self.encoder.classifier = proyector_MLP(out_dim, dimension)
        elif 'mobilenet_v2' in arch:
            out_dim = self.encoder.classifier[1].weight.shape[1]
            self.encoder.classifier = proyector_MLP(out_dim, dimension)
        elif 'alexnet' in arch:
            out_dim = self.encoder.classifier[1].weight.shape[1]
            self.encoder.classifier = proyector_MLP(out_dim, dimension)
        
        self.predictor = predictor_MLP(dimension)
    
    #Definimos las arquitecturas que vamos a implementar
    @staticmethod
    def get_backbone(backbone_name):
        return {'resnet34': resnet34(),
                'mobilenet_v2': mobilenet_v2(),
                'squeezenet1_1': squeezenet1_1(),
                'vgg13': vgg13(),
                'alexnet': alexnet()}[backbone_name]
    
    def forward(self, x1, x2):
        #Si es squeezenet tendremos que transformar los datos en un tensor de una columna
        if self.arch ==  'squeezenet1_1':
            x1 = self.encoder.features(x1)
            x2 = self.encoder.features(x2)
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            z1 = self.encoder.classifier(x1)
            z2 = self.encoder.classifier(x2)
        else:
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
            
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        return z1, z2, p1, p2







