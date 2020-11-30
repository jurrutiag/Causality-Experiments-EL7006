class ModelA2BC(nn.Module):
    def __init__(self, p_A, p_B_A, p_C_A):
        super().__init__()
        self.p_A = p_A
        self.p_B_A = p_B_A
        self.p_C_A = p_C_A

    def forward(self, inputs):
        # log P(B|A)P(C|A)P(A)
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        return self.p_A(inputs_A) + self.p_B_A(inputs_B, inputs_A) + self.p_C_A(inputs_C, inputs_A)

    def set_ground_truth(self, model, *args):
        if model.__name__ == self.__class__.__name__:
            p_A, p_B_A, p_C_A = [torch.log(torch.from_numpy(arg)) for arg in args]

            self.p_A.w.data = p_A
            self.p_B_A.w.data = p_B_A
            self.p_C_A.w.data = p_C_A

        else:
            log_joint = model.get_joint(*args)
            log_p_A = torch.logsumexp(log_joint, dim=(1, 2), keepdim=True)
            log_p_B_A = torch.logsumexp(log_joint - log_p_A, dim=2, keepdim=True)
            log_p_C_A = torch.logsumexp(log_joint - log_p_A, dim=1, keepdim=True)

            self.p_A.w.data = log_p_A.squeeze()
            self.p_B_A.w.data = log_p_B_A.squeeze()
            self.p_C_A.w.data = log_p_C_A.squeeze()


    def get_joint(*args):
        p_A, p_B_A, p_C_A = args

        p_A_th = torch.from_numpy(p_A).unsqueeze(1).unsqueeze(2)
        p_B_A_th = torch.from_numpy(p_B_A).unsqueeze(2)
        p_C_A_th = torch.from_numpy(p_C_A).unsqueeze(1)

        return torch.log(p_A_th) + torch.log(p_B_A_th) + torch.log(p_C_A_th)


class ModelB2AC(nn.Module):
    def __init__(self, p_B, p_A_B, p_C_B):
        super().__init__()
        self.p_B = p_B
        self.p_A_B = p_A_B
        self.p_C_B = p_C_B

    def forward(self, inputs):
        # log P(A|B)P(C|B)P(B)
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        return self.p_B(inputs_B) + self.p_A_B(inputs_A, inputs_B) + self.p_C_B(inputs_C, inputs_B)

    def set_ground_truth(self, model, *args):
        if model.__name__ == self.__class__.__name__:
            p_B, p_A_B, p_C_B = [torch.log(torch.from_numpy(arg)) for arg in args]

            self.p_B.w.data = p_B
            self.p_A_B.w.data = p_A_B
            self.p_C_B.w.data = p_C_B

        else:
            log_joint = model.get_joint(*args)
            log_p_B = torch.logsumexp(log_joint, dim=(0, 2), keepdim=True)
            log_p_A_B = torch.logsumexp(log_joint - log_p_B, dim=2, keepdim=True)
            log_p_C_B = torch.logsumexp(log_joint - log_p_B, dim=0, keepdim=True)

            self.p_B.w.data = log_p_B.squeeze()
            self.p_A_B.w.data = log_p_A_B.squeeze()
            self.p_C_B.w.data = log_p_C_B.squeeze()


    def get_joint(*args):
        p_B, p_A_B, p_C_B = args

        p_B_th = torch.from_numpy(p_B).unsqueeze(0).unsqueeze(2)
        p_A_B_th = torch.from_numpy(p_A_B).unsqueeze(2)
        p_C_B_th = torch.from_numpy(p_C_B).unsqueeze(0)

        return torch.log(p_B_th) + torch.log(p_A_B_th) + torch.log(p_C_B_th)

class ModelC2AB(nn.Module):
    def __init__(self, p_C, p_A_C, p_B_C):
        super().__init__()
        self.p_C = p_C
        self.p_A_C = p_A_C
        self.p_B_C = p_B_C

    def forward(self, inputs):
        # log P(A|C)P(B|C)P(C)
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        return self.p_C(inputs_C) + self.p_A_C(inputs_A, inputs_C) + self.p_B_C(inputs_B, inputs_C)

    def set_ground_truth(self, model, *args):
        if model.__name__ == self.__class__.__name__:
            p_C, p_A_C, p_B_C = [torch.log(torch.from_numpy(arg)) for arg in args]

            self.p_C.w.data = p_C
            self.p_A_C.w.data = p_A_C
            self.p_B_C.w.data = p_B_C

        else:
            log_joint = model.get_joint(*args)
            log_p_C = torch.logsumexp(log_joint, dim=(0, 1), keepdim=True)
            log_p_A_C = torch.logsumexp(log_joint - log_p_C, dim=1, keepdim=True)
            log_p_B_C = torch.logsumexp(log_joint - log_p_C, dim=0, keepdim=True)

            self.p_C.w.data = log_p_C.squeeze()
            self.p_A_C.w.data = log_p_A_C.squeeze()
            self.p_B_C.w.data = log_p_B_C.squeeze()


    def get_joint(*args):
        p_C, p_A_C, p_B_C = args

        p_C_th = torch.from_numpy(p_C).unsqueeze(0).unsqueeze(1)
        p_A_C_th = torch.from_numpy(p_A_C).unsqueeze(1)
        p_B_C_th = torch.from_numpy(p_B_C).unsqueeze(0)

        return torch.log(p_C_th) + torch.log(p_A_C_th) + torch.log(p_B_C_th)

class ModelA2BC_B2C(nn.Module):
    def __init__(self, p_A, p_C_AB, p_B_A):
        super().__init__()
        self.p_A = p_A
        self.p_C_AB = p_C_AB
        self.p_B_A = p_B_A

    def forward(self, inputs):
        # P(C|A, B)P(B|A)P(A)
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        return self.p_A(inputs_A) + self.p_C_AB(inputs_C, inputs_A, inputs_B) + self.p_B_A(inputs_B, inputs_A)

    def set_ground_truth(self, model, *args):
        if model.__name__ == self.__class__.__name__:
            p_A, p_C_AB, p_B_A = [torch.log(torch.from_numpy(arg)) for arg in args]

            self.p_A.w.data = p_A
            self.p_C_AB.w.data = p_C_AB
            self.p_B_A.w.data = p_B_A

        else:
            log_joint = model.get_joint(*args)
            log_p_A = torch.logsumexp(log_joint, dim=(1, 2), keepdim=True)
            log_p_B = torch.logsumexp(log_joint, dim=(0, 2), keepdim=True)
            log_p_C_AB = log_joint - log_p_A - log_p_B
            log_p_B_A = torch.logsumexp(log_joint - log_p_A, dim=2, keepdim=True)

            self.p_A.w.data = log_p_A.squeeze()
            self.p_C_AB.w.data = log_p_C_AB.squeeze()
            self.p_B_A.w.data = log_p_B_A.squeeze()


    def get_joint(*args):
        p_A, p_C_AB, p_B_A = args

        p_A_th = torch.from_numpy(p_A).unsqueeze(1).unsqueeze(2)
        p_C_AB_th = torch.from_numpy(p_C_AB)
        p_B_A_th = torch.from_numpy(p_B_A).unsqueeze(2)

        return torch.log(p_A_th) + torch.log(p_C_AB_th) + torch.log(p_B_A_th)

class ModelA2BC_C2B(nn.Module):
    def __init__(self, p_A, p_B_AC, p_C_A):
        super().__init__()
        self.p_A = p_A
        self.p_B_AC = p_B_AC
        self.p_C_A = p_C_A

    def forward(self, inputs):
        # P(B|A, C)P(C|A)P(A)
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        return self.p_A(inputs_A) + self.p_B_AC(inputs_B, inputs_A, inputs_C) + self.p_C_A(inputs_C, inputs_A)

    def set_ground_truth(self, model, *args):
        if model.__name__ == self.__class__.__name__:
            p_A, p_B_AC, p_C_A = [torch.log(torch.from_numpy(arg)) for arg in args]

            self.p_A.w.data = p_A
            self.p_B_AC.w.data = p_B_AC
            self.p_C_A.w.data = p_C_A

        else:
            log_joint = model.get_joint(*args)
            log_p_A = torch.logsumexp(log_joint, dim=(1, 2), keepdim=True)
            log_p_C = torch.logsumexp(log_joint, dim=(0, 1), keepdim=True)
            log_p_B_AC = log_joint - log_p_A - log_p_C
            log_p_C_A = torch.logsumexp(log_joint - log_p_A, dim=1, keepdim=True)

            self.p_A.w.data = log_p_A.squeeze()
            self.p_B_AC.w.data = log_p_B_AC.squeeze()
            self.p_C_A.w.data = log_p_C_A.squeeze()


    def get_joint(*args):
        p_A, p_B_AC, p_C_A = args

        p_A_th = torch.from_numpy(p_A).unsqueeze(1).unsqueeze(2)
        p_B_AC_th = torch.from_numpy(p_B_AC)
        p_C_A_th = torch.from_numpy(p_C_A).unsqueeze(2)

        return torch.log(p_A_th) + torch.log(p_B_AC_th) + torch.log(p_C_A_th)

class ModelB2AC_A2C(nn.Module):
    def __init__(self, p_B, p_C_AB, p_A_B):
        super().__init__()
        self.p_B = p_B
        self.p_C_AB = p_C_AB
        self.p_A_B = p_A_B

    def forward(self, inputs):
        # P(C|A, B)P(A|B)P(B)
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        return self.p_B(inputs_B) + self.p_C_AB(inputs_C, inputs_A, inputs_B) + self.p_A_B(inputs_A, inputs_B)

    def set_ground_truth(self, model, *args):
        if model.__name__ == self.__class__.__name__:
            p_B, p_C_AB, p_A_B = [torch.log(torch.from_numpy(arg)) for arg in args]

            self.p_B.w.data = p_B
            self.p_C_AB.w.data = p_C_AB
            self.p_A_B.w.data = p_A_B

        else:
            log_joint = model.get_joint(*args)
            log_p_A = torch.logsumexp(log_joint, dim=(1, 2), keepdim=True)
            log_p_B = torch.logsumexp(log_joint, dim=(0, 2), keepdim=True)
            log_p_C_AB = log_joint - log_p_A - log_p_B
            log_p_A_B = torch.logsumexp(log_joint - log_p_B, dim=2, keepdim=True)

            self.p_B.w.data = log_p_B.squeeze()
            self.p_C_AB.w.data = log_p_C_AB.squeeze()
            self.p_A_B.w.data = log_p_A_B.squeeze()


    def get_joint(*args):
        p_B, p_C_AB, p_A_B = args

        p_B_th = torch.from_numpy(p_B).unsqueeze(1).unsqueeze(2)
        p_C_AB_th = torch.from_numpy(p_C_AB)
        p_A_B_th = torch.from_numpy(p_A_B).unsqueeze(2)

        return torch.log(p_B_th) + torch.log(p_C_AB_th) + torch.log(p_A_B_th)
        

class ModelB2AC_C2A(nn.Module):
    def __init__(self, p_B, p_A_CB, p_C_B):
        super().__init__()
        self.p_B = p_B
        self.p_A_CB = p_A_CB
        self.p_C_B = p_C_B

    def forward(self, inputs):
        # P(A|C, B)P(C|B)P(B)
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        return self.p_B(inputs_B) + self.p_A_CB(inputs_A, inputs_C, inputs_B) + self.p_C_B(inputs_C, inputs_B)

    def set_ground_truth(self, model, *args):
        if model.__name__ == self.__class__.__name__:
            p_B, p_A_CB, p_C_B = [torch.log(torch.from_numpy(arg)) for arg in args]

            self.p_B.w.data = p_B
            self.p_A_CB.w.data = p_A_CB
            self.p_C_B.w.data = p_C_B

        else:
            log_joint = model.get_joint(*args)
            log_p_C = torch.logsumexp(log_joint, dim=(0, 1), keepdim=True)
            log_p_B = torch.logsumexp(log_joint, dim=(0, 2), keepdim=True)
            log_p_A_CB = log_joint - log_p_C - log_p_B
            log_p_C_B = torch.logsumexp(log_joint - log_p_B, dim=0, keepdim=True)

            self.p_B.w.data = log_p_B.squeeze()
            self.p_A_CB.w.data = log_p_A_CB.squeeze()
            self.p_C_B.w.data = log_p_C_B.squeeze()


    def get_joint(*args):
        p_B, p_A_CB, p_C_B = args

        p_B_th = torch.from_numpy(p_B).unsqueeze(1).unsqueeze(2)
        p_A_CB_th = torch.from_numpy(p_A_CB)
        p_C_B_th = torch.from_numpy(p_C_B).unsqueeze(2)

        return torch.log(p_B_th) + torch.log(p_A_CB_th) + torch.log(p_C_B_th)


class ModelC2AB_A2B(nn.Module):
    def __init__(self, p_C, p_B_AC, p_A_C):
        super().__init__()
        self.p_C = p_C
        self.p_B_AC = p_B_AC
        self.p_A_C = p_A_C

    def forward(self, inputs):
        # P(B|A, C)P(A|C)P(C)
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        return self.p_C(inputs_C) + self.p_B_AC(inputs_B, inputs_A, inputs_C) + self.p_A_C(inputs_A, inputs_C)

    def set_ground_truth(self, model, *args):
        if model.__name__ == self.__class__.__name__:
            p_C, p_B_AC, p_A_C = [torch.log(torch.from_numpy(arg)) for arg in args]

            self.p_C.w.data = p_C
            self.p_B_AC.w.data = p_B_AC
            self.p_A_C.w.data = p_A_C

        else:
            log_joint = model.get_joint(*args)
            log_p_C = torch.logsumexp(log_joint, dim=(0, 1), keepdim=True)
            log_p_A = torch.logsumexp(log_joint, dim=(1, 2), keepdim=True)
            log_p_B_AC = log_joint - log_p_C - log_p_A
            log_p_A_C = torch.logsumexp(log_joint - log_p_C, dim=1, keepdim=True)

            self.p_C.w.data = log_p_C.squeeze()
            self.p_B_AC.w.data = log_p_B_AC.squeeze()
            self.p_A_C.w.data = log_p_A_C.squeeze()


    def get_joint(*args):
        p_C, p_B_AC, p_A_C = args

        p_C_th = torch.from_numpy(p_C).unsqueeze(1).unsqueeze(2)
        p_B_AC_th = torch.from_numpy(p_B_AC)
        p_A_C_th = torch.from_numpy(p_A_C).unsqueeze(2)

        return torch.log(p_C_th) + torch.log(p_B_AC_th) + torch.log(p_A_C_th)


class ModelC2AB_B2A(nn.Module):
    def __init__(self, p_C, p_A_BC, p_B_C):
        super().__init__()
        self.p_C = p_C
        self.p_A_BC = p_A_BC
        self.p_B_C = p_B_C

    def forward(self, inputs):
        # P(A|B, C)P(B|C)P(C)
        inputs_A, inputs_B, inputs_C = torch.split(inputs, 1, dim=1)
        return self.p_C(inputs_C) + self.p_A_BC(inputs_A, inputs_B, inputs_C) + self.p_B_C(inputs_B, inputs_C)

    def set_ground_truth(self, model, *args):
        if model.__name__ == self.__class__.__name__:
            p_C, p_A_BC, p_B_C = [torch.log(torch.from_numpy(arg)) for arg in args]

            self.p_C.w.data = p_C
            self.p_A_BC.w.data = p_A_BC
            self.p_B_C.w.data = p_B_C

        else:
            log_joint = model.get_joint(*args)
            log_p_C = torch.logsumexp(log_joint, dim=(0, 1), keepdim=True)
            log_p_B = torch.logsumexp(log_joint, dim=(0, 2), keepdim=True)
            log_p_A_BC = log_joint - log_p_C - log_p_B
            log_p_B_C = torch.logsumexp(log_joint - log_p_C, dim=0, keepdim=True)

            self.p_C.w.data = log_p_C.squeeze()
            self.p_A_BC.w.data = log_p_A_BC.squeeze()
            self.p_B_C.w.data = log_p_B_C.squeeze()


    def get_joint(*args):
        p_C, p_A_BC, p_B_C = args

        p_C_th = torch.from_numpy(p_C).unsqueeze(1).unsqueeze(2)
        p_A_BC_th = torch.from_numpy(p_A_BC)
        p_B_C_th = torch.from_numpy(p_B_C).unsqueeze(2)

        return torch.log(p_C_th) + torch.log(p_A_BC_th) + torch.log(p_B_C_th)
