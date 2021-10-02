import sys
import torch
import torch.nn as nn
import torch.optim as optim
import os
from odak.learn.wave import wavenumber,generate_complex_field,calculate_amplitude,propagate_beam,linear_grating,calculate_phase
from odak.learn.tools import zero_pad,crop_center,save_image
from odak.tools import check_directory
from odak import np
from data import DatasetFromFolder,load
from tqdm import tqdm

sys.path.append('../loss_functions')

def prepare(settings,wavelength):
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    cuda          = settings["general"]["cuda"]
    resolution    = settings["slm"]["resolution"]
    device        = torch.device("cuda" if cuda else "cpu")
    if cuda:
       torch.cuda.empty_cache()
    torch.random.seed()
    kernel        = torch.rand(
                               1,
                               2,
                               resolution[0],
                               resolution[1],
                              ).detach().to(device).requires_grad_()
    dataset       = DatasetFromFolder(
                                      settings["dataset"]["input directory"],
                                      settings["dataset"]["output directory"],
                                      device
                                     )
    target        = load(settings["general"]["target filename"],device)
    criterion     = [
                     nn.MSELoss().to(device),
                    ]
    return kernel,target,dataset,criterion,device 

def evaluate(image,target,criterion,w=[1.,]):
    loss   = w[0]*criterion[0](image,target)
    return loss

def optimize(settings,wavelength,kernel,target,criterion,device,multiplier=1.0):
    image_location       = settings["image"]["location"][2]
    pixel_pitch          = settings["slm"]["pixel pitch"]
    resolution           = settings["slm"]["resolution"]
    propagation_type     = settings["general"]["propagation type"]
    loss_weights         = settings["general"]["loss weights"]
    learning_rate        = settings["general"]["learning rate"]
    n_iterations         = settings["general"]["iterations"]
    m                    = settings["general"]["region of interest"]
    ones                 = torch.ones(resolution[0],resolution[1],requires_grad=False).to(device)
    input_phase          = torch.rand(resolution[0],resolution[1]).detach().to(device).requires_grad_()
    optimizer            = optim.Adam([{'params': [input_phase]}],lr=learning_rate)
    t                    = tqdm(range(n_iterations),leave=False)
    mask                 = torch.zeros(resolution[0],resolution[1],requires_grad=False).to(device)
    mask[
         int(resolution[0]*m[0]):int(resolution[0]*m[1]),
         int(resolution[1]*m[0]):int(resolution[1]*m[1])
        ]                = 1
    if type(kernel) != type(None):
        kernel.requires_grad = False
    for n in t:
        optimizer.zero_grad()
        field       = a_single_step(
                                    ones,
                                    input_phase,
                                    kernel,
                                    image_location,
                                    wavelength,
                                    pixel_pitch,
                                    propagation_type
                                   )
        image       = calculate_amplitude(field)**2
        loss        = evaluate(image*mask,target*mask*multiplier,criterion,w=loss_weights)
        description = "Iteration:{}, Loss:{:.4f}".format(n,loss.item()) 
        loss.backward(retain_graph=True)
        optimizer.step()
        t.set_description(description)
    if 'description'  in locals():
       print(description)
    torch.cuda.empty_cache()
    return input_phase.detach(),image.detach()

def find_kernel(settings,wavelength,kernel,dataset,criterion,device):
    image_location   = settings["image"]["location"][2]
    pixel_pitch      = settings["slm"]["pixel pitch"]
    resolution       = settings["slm"]["resolution"]
    loss_weights     = settings["general"]["loss weights"]
    propagation_type = settings["general"]["propagation type"]
    m                = settings["kernel"]["region of interest"]
    learning_rate    = settings["kernel"]["learning rate"]
    n_iterations     = settings["kernel"]["iterations"]
    optimizer        = optim.Adam([{'params': [kernel]}],lr=learning_rate)
    ones             = torch.ones(resolution[0],resolution[1],requires_grad=False).to(device)
    mask             = torch.zeros(resolution[0],resolution[1],requires_grad=False).to(device)
    mask[
         int(resolution[0]*m[0]):int(resolution[0]*m[1]),
         int(resolution[1]*m[0]):int(resolution[1]*m[1])
        ]              = 1
    t     = tqdm(range(n_iterations),leave=False)
    for n in t:
        total_loss = 0 
        id_set     = range(dataset.__len__())
        t0         = tqdm(range(dataset.__len__()),leave=False)
        for i in t0:
            optimizer.zero_grad()
            input_phase,target = dataset.__getitem__(i)
            input_phase        = input_phase*2*np.pi
            field              = a_single_step(
                                               ones,
                                               input_phase,
                                               kernel,
                                               image_location,
                                               wavelength,
                                               pixel_pitch,
                                               propagation_type
                                              )
            image              = calculate_amplitude(field)**2
            loss               = evaluate(image*mask,target*mask,criterion)
            total_loss        += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            description = "Image:{}, Loss:{:.4f}".format(i,loss.item()) 
            t0.set_description(description)
            if n == n_iterations-1:
                if i == 0:
                    save_multiple(n,i,input_phase,target,image,settings)
        total_loss  /= dataset.__len__()
        description  = "Iteration:{}, Loss:{:.4f}".format(n,total_loss) 
        t.set_description(description)
    if 'description'  in locals():
        print(description)
    torch.cuda.empty_cache()
    return kernel

def save_multiple(n,i,input_phase,target,image,settings):
    save(
         input_phase,
         'input_phase_{}_{}.png'.format(n,i),
         directory=settings["general"]["output directory"],
         save_type='phase'
        )
    save(
         target*255.,
         'output_{}_{}.png'.format(n,i),
         directory=settings["general"]["output directory"],
         save_type='image'
        )
    save(
         image*255.,
         'reconstruction_{}_{}.png'.format(n,i),
         directory=settings["general"]["output directory"],
         save_type='image'
        )

def a_single_step(hologram_amplitude,hologram_phase,kernel,distance,wavelength,pixel_pitch,propagation_type):
    field        = generate_complex_field(hologram_amplitude,hologram_phase) 
    k            = wavenumber(wavelength)
    field_padded = zero_pad(field) 
    if type(kernel) == type(None):
        final_field_padded = propagate_beam(field_padded,k,distance,pixel_pitch,wavelength,propagation_type)
        final_field        = crop_center(final_field_padded)
        return final_field
    final_field = torch.zeros((kernel.shape[0],kernel.shape[2],kernel.shape[3]),dtype=torch.complex64).to(field.device)
    h_a              = kernel[0,0]
    h_p              = kernel[0,1]
    h                = generate_complex_field(h_a,h_p)
    h                = zero_pad(h)
    new_field_padded = propagate_beam(field_padded,None,None,None,None,propagation_type='custom',kernel=h)
    new_field        = crop_center(new_field_padded)
    return_field     = new_field
    return return_field

def single_propagation(field,H):
    U1     = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field)))
    U2     = H*U1
    result = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(U2)))
    return result

def save(field,filename='output.png',directory='./',save_type='image'):
    check_directory(directory)
    fn = '{}/{}'.format(directory,filename)
    if save_type == 'image':
        field_save = (field-field.min())/(field.max()-field.min())*255.
    elif save_type == 'phase':
        field_save = field%(2*np.pi)/(2*np.pi)*255.
    save_image(fn,field_save)

def start(settings):
    wavelength = settings["beam"]["wavelength"]
    directory  = settings["general"]["output directory"]
    multiplier = settings["kernel"]["multiplier"]
    fn_kernel  = './calibrations/kernel.pt'
    kernel,target,dataset,criterion,device = prepare(settings,wavelength)
    check_directory('./calibrations')
    if os.path.isfile(fn_kernel) == True:
        kernel = torch.load(fn_kernel).to(device)
        settings["kernel"]["iterations"] = 0
    kernel                         = find_kernel(
                                                 settings,
                                                 wavelength,
                                                 kernel,
                                                 dataset,
                                                 criterion,
                                                 device,
                                                )
    torch.save(kernel,fn_kernel)
    checker_complex                = linear_grating(
                                                    settings["slm"]["resolution"][0],
                                                    settings["slm"]["resolution"][1],
                                                    axis='y'
                                                   ).to(device)
    checker                       = calculate_phase(checker_complex)
    hologram_phase,reconstruction = optimize(
                                             settings,
                                             wavelength,
                                             kernel,
                                             target,
                                             criterion,
                                             device,
                                             multiplier
                                            )
    save(reconstruction,filename='reconstruction.png',directory=directory,save_type='image')
    save(hologram_phase,filename='hologram_phase.png',directory=directory,save_type='phase')
    save(hologram_phase+checker,filename='hologram_phase_checker.png',directory=directory,save_type='phase')
    save(target,filename='target.png',directory=directory,save_type='image')
    multiplier                                = settings["ideal"]["multiplier"]
    hologram_phase_ideal,reconstruction_ideal = optimize(
                                                         settings,
                                                         wavelength,
                                                         None,
                                                         target,
                                                         criterion,
                                                         device,
                                                         multiplier
                                                        )
    save(reconstruction_ideal,filename='reconstruction_ideal.png',directory=directory,save_type='image')
    save(hologram_phase_ideal,filename='hologram_phase_ideal.png',directory=directory,save_type='phase')
    save(hologram_phase_ideal+checker,filename='hologram_phase_ideal_checker.png',directory=directory,save_type='phase')
    save(target,filename='target.png',directory=directory,save_type='image')
    return True
