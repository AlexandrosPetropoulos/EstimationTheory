
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from matplotlib.animation import FuncAnimation

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from scipy import ndimage

import os
class ExtendedKalmanFilter:
    
    def __init__(self,x0 = None ,P = None,motion_var = None, measurement_var = None, Q = None, R = None , dt = None):
        
        if(x0 is None):
            self.xt = np.matrix([0,0,0,0,0,0,0]).T
        else:
            self.xt = x0
            
        if(dt is None):
            self.dt=0.1
        else:
            self.dt=dt

        #initial covariance matrix of gaussian distribution
        if(P is None):
            self.P = np.matrix([[0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0],
                                [0,0,0,10,0,0,0],
                                [0,0,0,0,10,0,0],
                                [0,0,0,0,0,10,0],
                                [0,0,0,0,0,0,10]])
        else:
            self.P = P
         
        #matrix to convert 3x3 to 7x7
        self.Fx = np.matrix([[1,0,0,0,0,0,0],
                             [0,1,0,0,0,0,0],
                             [0,0,1,0,0,0,0]])
        
        if(motion_var is None):
            self.motion_var = 0.01
        else:
            self.motion_var = motion_var
            
        #gia tin kinisi, einai 3x3 kai me ton Fx tha paei sto 7x7
        if(Q is None):
            temp = np.array([[self.motion_var**2,0,0],
                             [0,self.motion_var**2,0],
                             [0,0,np.deg2rad(30)**2]])
            self.Q = np.asmatrix(temp)
        else:
            self.Q = Q
            
        if(measurement_var is None):
            self.measurement_var = 0.5
        else:
            self.measurement_var = measurement_var
            
        if(R is None):
            temp = np.identity(2)*(0.5**2)
            temp[1][1] = (0.3**2)
            self.R = np.asmatrix(temp)
        else:
            self.R  = np.identity(2)*(self.measurement_var)
            
        self.Fxj1 = np.matrix([[1,0,0,0,0,0,0],
		                       [0,1,0,0,0,0,0],
		                       [0,0,1,0,0,0,0],
		                       [0,0,0,1,0,0,0],
		                       [0,0,0,0,1,0,0]])

        self.Fxj2 = np.matrix([[1,0,0,0,0,0,0],
		                       [0,1,0,0,0,0,0],
		                       [0,0,1,0,0,0,0],
		                       [0,0,0,0,0,1,0],
		                       [0,0,0,0,0,0,1]])
    
        self.Fxj = [self.Fxj1,self.Fxj2]
            
        self.landmarks_first_seen = 1
        #voithitikes metavlites p.x. gia na kratao ola ta P apo toin arxi os to telos
        self.x = []
        self.PP = []

    def __fxu_at(self, ut):
        
        theta = self.xt.item(2)
        
        v = ut[0]
        w = ut[1]
        
        temp = np.zeros((3,1))
        
        #vlepe sxolio sto predict , giati den vazo ta proigoumena x, y kai theta stin eksisosi
        temp[0][0] = self.dt*v*np.cos(theta)
        temp[1][0] = self.dt*v*np.sin(theta)
        temp[2][0] = self.dt*w


        return np.matrix(temp).astype(float)
    
    #jacobian tis fxu_at
    def __F_jacobian_at(self, ut):
        
        temp = np.identity(3)
    
        temp[0][2] = (-self.dt)*ut[0]*np.sin(self.xt[2])
        temp[1][2] = (self.dt)*ut[0]*np.cos(self.xt[2])
    
        temp[0][0] = 0
        temp[1][1] = 0
        temp[2][2] = 0

        return np.matrix(temp).astype(float)
    
    def __Ht_jacobian_at(self,pp):
    
        x = self.xt.item(0)
        y = self.xt.item(1)
        X1 = self.xt.item(3)
        Y1 = self.xt.item(4)
        X2 = self.xt.item(5)
        Y2 = self.xt.item(6)
        if(pp == 1):
            X1 = X2
            Y1 = Y2
        
        q = np.power(x-X1,2) + np.power(y-Y1,2)

        temp = np.zeros((2,5)).astype(float)
        
        temp[0][0] = (-X1+x)/np.sqrt(q)
        temp[0][1] = (-Y1+y)/np.sqrt(q)
        temp[0][3] = (X1-x)/np.sqrt(q)
        temp[0][4] = (Y1-y)/np.sqrt(q)
        
        temp[1][0] = -(-Y1+y)/q
        temp[1][1] = -(X1-x)/q
        temp[1][2] = -1
        temp[1][3] = (-Y1+y)/q
        temp[1][4] = (X1-x)/q
        

    
        return np.matrix(temp).astype(float)
         

    def predict(self, ut):

        self.xt = self.xt + self.Fx.T*self.__fxu_at(ut)
        
        #i kainourgia jacobian vgenei apo tin palia pou einai diastaseon 3x3
        __F_jacobian_at = np.asmatrix(np.identity(7)) + self.Fx.T*self.__F_jacobian_at(ut)*self.Fx

        #ensure angle is between 0,2pi
        self.xt[2] = pi2pi(self.xt[2])
        self.P = __F_jacobian_at*self.P*__F_jacobian_at.T + self.Q#self.Fx.T*self.Q*self.Fx



        return self.xt,self.P

    def update(self, z):
      
        #akomi kai na kano arxikopoiisi sta x,y ton landmark, epeidi trexo to predict
        #ta midenizei, opote vazo edo tin proti ektimisi tis thesis ton landmark
        #vazo ta x kai y, giati exoun allaksei apo to predict kai den einai 0, opos stin arxikopoiisi
        if(self.landmarks_first_seen == 1):
            x = self.xt.item(0)
            y = self.xt.item(1)
            #theta = self.xt.item(2)
            self.xt[3] = x + z.item(0)*np.cos(z.item(1))
            self.xt[4] = y + z.item(0)*np.sin(z.item(1))
            self.xt[5] = x + z.item(2)*np.cos(z.item(3))
            self.xt[6] = y + z.item(2)*np.sin(z.item(3))
            self.landmarks_first_seen = 0
          
        #pragmatiki metrisi
        z_real = [z[0:2], z[2:4]]
        
        #gia kathe empodio
        for t in range(2): 
            x = self.xt.item(0)
            y = self.xt.item(1)
            theta = pi2pi(self.xt.item(2)) 
            X1 = self.xt.item(3)
            Y1 = self.xt.item(4)

            if(t==1):
                X1 = self.xt.item(5) # diladi = X2
                Y1 = self.xt.item(6) # diladi = Y2

            
            q = np.power(x-X1,2) + np.power(y-Y1,2)
           
            zt_est = np.matrix([[np.sqrt(q)],[pi2pi(np.arctan2(Y1-y,X1-x)-theta)]])
        
   
            Hti = self.__Ht_jacobian_at(t)*self.Fxj[t]
            
            K = self.P*Hti.T*(Hti*self.P*Hti.T + self.R).I 

            tmp = (z_real[t]-zt_est)
            tmp[1]=pi2pi(tmp[1])

            self.xt = self.xt + K*(tmp)
            self.P = (np.matrix(np.identity(7)) - K*Hti)*self.P
            

        self.x.append(self.xt)
        self.PP.append(self.P)

        return self.xt,self.P
    
def pi2pi(a):
#    if(a<0):
#        a = a % (2 * np.pi) * (-1)
#    else:
    a = a % (2 * np.pi)
    
    if a > np.pi:             # move to [-pi, pi)
        a -= 2 * np.pi
        
    return a


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]       

def plotEllipse(axis,x,y,P,std,facecolor,alpha):
    #https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
    #http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    nstd = std
    
    cov = P

    vals, vecs = eigsorted(cov)
    vals = vals[0:2]
    vecs = vecs[0:2]
    
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=(x, y),
              width=w, height=h,
              angle=theta, color='black')
    ell.set_facecolor(facecolor)
    ell.set_alpha(alpha)
    axis.add_artist(ell)
    
    return ell
    
    
    
    
def example():

    #dt=0.1
    
    df1 = pd.read_csv("dataset/control1.csv", header=None )
    df1 = np.array(df1)
    
    df2 = pd.read_csv("dataset/radar1.csv", header=None )
    df2 = np.array(df2)
    
    #Q = np.matrix([[2.5e-06,0,0],[0,1.0e-03,0],[0,0,1.0e-01]])
    Q = np.matrix([[0.01**2,0,0,0,0,0,0],
                   [0,0.01**2,0,0,0,0,0],
                   [0,0,0.01**2,0,0,0,0],
                   [0,0,0,0.001,0,0,0],
                   [0,0,0,0,0.001,0,0],
                   [0,0,0,0,0,0.001,0],
                   [0,0,0,0,0,0,0.001]])


    ek = ExtendedKalmanFilter( Q=Q,measurement_var = 0.5, dt = 0.1)

    fig = plt.figure( )
    axRobotPosition = plt.subplot(111) 
    axRobotPosition.set_title('Kalman filter with static landmarks')
    axRobotPosition.set_xlim(-6, 6)
    axRobotPosition.set_ylim(-1, 9)


    temp_x = []
    temp_y = []
    for i in range(df1.shape[0]):#df1.shape[0]
    
        ek.predict(list(df1[i]))
        #print(ek.xt.item(2))
        zt = np.matrix([[df2[i][0]],[df2[i][1]],[df2[i][2]],[df2[i][3]]])
        ek.update(zt)
        temp_x.append(ek.xt.item(0))
        temp_y.append(ek.xt.item(1))
        #print(ek.xt.item(2))
        lambda_, v = np.linalg.eig(ek.P[0:2, 0:2])
        lambda_ = np.sqrt(lambda_)
        print(lambda_)
        #old plot
        #axRobotPosition.clear()
    #    axRobotPosition.plot(ek.xt.item(0), ek.xt.item(1), marker=(3, 0, math.degrees(ek.xt.item(2))),color='green', markersize=20, linestyle='None')
    #    axRobotPosition.plot(ek.xt.item(0), ek.xt.item(1), 'bo')
    #    axRobotPosition.quiver(ek.xt.item(0), ek.xt.item(1),np.cos(ek.xt.item(2)), np.sin(ek.xt.item(2)),linewidths=0.01, edgecolors='k')
    #    axRobotPosition.quiver(ek.xt.item(0), ek.xt.item(1),ek.xt.item(0), ek.xt.item(1),linewidths=0.01, edgecolors='k')
    #    axRobotPosition.scatter(ek.xt.item(0), ek.xt.item(1), s=20, c = 'b')
    #    axRobotPosition.plot(temp_x,temp_y,color='blue')
    
        #arxikopoiisis ton text
        if(i==0):
            text_No_iteration = axRobotPosition.text(-5.5, -0.5, f'Iteration: {i}', style='italic',bbox={'facecolor': 'red', 'alpha': 0.1, 'pad': 2})
            #t10 = axRobotPosition.text(-1.5, 8, f'Radar distance is: {df2[i,0]}', style='italic',bbox={'facecolor': 'green', 'alpha': 0.1, 'pad': 2})
            #t8 = axRadar.text(1, -0.5, f'Iteration: {i}', style='italic',bbox={'facecolor': 'red', 'alpha': 0.1, 'pad': 2})
            #t9 = axRadar.text(1, 8, f'Radar distance is: {df2[i,0]}', style='italic',bbox={'facecolor': 'green', 'alpha': 0.1, 'pad': 2})
        text_No_iteration.set_text(f'Iteration: {i}')
        #t8.set_text(f'Iteration: {i}')
        #t9.set_text(f'Radar distance is: {df2[i,0]}')
        #t10.set_text(f'estimated distance is: {round(np.sqrt((ek.xt.item(0)-ek.xt.item(3))**2+(ek.xt.item(1)-ek.xt.item(4))**2),3)}')
        #line_between_points = axRobotPosition.plot([ek.xt.item(0),ek.xt.item(3)],[ek.xt.item(1),ek.xt.item(4)],'-',color = 'saddlebrown')#'k-',
        axRobotPosition.grid(True)
        #robot_ellipse = plotEllipse(axRobotPosition,ek.xt.item(0), ek.xt.item(1),ek.P[0:2, 0:2],std = 1, facecolor='lightskyblue',alpha = 0.1)#if (i%5 == 0 ):    
        land1_ellipse = plotEllipse(axRobotPosition,ek.xt.item(3), ek.xt.item(4),ek.P[3:5, 3:5],std = 1, facecolor='lightcoral',alpha = 0.1)
        land2_ellipse = plotEllipse(axRobotPosition,ek.xt.item(5), ek.xt.item(6),ek.P[5:7, 5:7],std = 1, facecolor='lightgreen',alpha = 0.1)
        #plot_covariance_ellipse((ek.xt[0,0], ek.xt[1,0]), ek.P[0:2, 0:2],std=1, facecolor='lightskyblue', alpha=0.05)
        #plot_covariance_ellipse((ek.xt[3,0], ek.xt[4,0]), ek.P[3:5, 3:5],std=1, facecolor='lightcoral', alpha=0.05)
        #plot_covariance_ellipse((ek.xt[5,0], ek.xt[6,0]), ek.P[5:7, 5:7],std=1, facecolor='lightgreen', alpha=0.05)
        axRobotPosition.scatter(ek.xt.item(0), ek.xt.item(1), s=20, c = 'b')
        
        image_path = get_sample_data('C:\\Users\\alex\\Documents\\Estimation Theory\\car.png')
        car_plot = imscatter(ek.xt.item(0), ek.xt.item(1), image_path,np.degrees(ek.xt.item(2)), zoom=0.1, ax=axRobotPosition)
        
        robot_quiver = axRobotPosition.quiver(ek.xt.item(0), ek.xt.item(1),np.cos(ek.xt.item(2)), np.sin(ek.xt.item(2)),scale=11,linewidths=0.01, edgecolors='k')
        axRobotPosition.plot(ek.xt.item(3), ek.xt.item(4), 'rx')
        #plotEllipse(axRobotPosition,ek.xt.item(3), ek.xt.item(4),ek.P)
        axRobotPosition.plot(ek.xt.item(5), ek.xt.item(6), 'gx')
        #axRobotPosition.legend(('robot', 'landmark1', 'landmark2'))
        robot_ellipse = plotEllipse(axRobotPosition,ek.xt.item(0), ek.xt.item(1),ek.P[0:2, 0:2],std = 1, facecolor='lightskyblue',alpha = 0.1)
        axRobotPosition.relim()
        axRobotPosition.autoscale_view()
        #axRobotPosition.set_aspect('equal')
        
        #remove the previous plotted object
        #einai aparaitito to pause, episis an ginoun meta to pause kanei spasimata
        if(i==0):#gia to capture tou video
            plt.pause(2)
        plt.pause(0.05)

        # filename = "plot" + str(i) + ".png"
        # fig.savefig(os.path.join("dataset/", filename), bbox_inches='tight')

        robot_quiver.set_visible(False)
        if(i>5):
            robot_ellipse.set_visible(False)
            land1_ellipse.set_visible(False)
            land2_ellipse.set_visible(False)

        car_plot.pop(0).remove()

           

        #https://stackoverflow.com/questions/51133678/matplotlib-figure-flickering-when-updating
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        #plt.pause(1)




    return ek.xt,ek.P,ek.x,ek.PP


#def animated_plot(i):
#    global x
#    global PP
#    
#    #fig = plt.figure()
##    axRobotPosition = plt.subplot(111) 
#    axRobotPosition.set_xlim(-6, 6)
#    axRobotPosition.set_ylim(-1, 9)
#    
#    #for i in range(df1.shape[0]):
#    axRobotPosition.grid(True)
#    t3 = plotEllipse(axRobotPosition,x[i].item(0), x[i].item(1),PP[i][0:2, 0:2],std = 1, facecolor='lightskyblue',alpha = 0.05)#if (i%5 == 0 ):    
#    t4 = plotEllipse(axRobotPosition,x[i].item(3), x[i].item(4),PP[i][3:5, 3:5],std = 1, facecolor='lightcoral',alpha = 0.05)
#    t5 = plotEllipse(axRobotPosition,x[i].item(5), x[i].item(6),PP[i][5:7, 5:7],std = 1, facecolor='lightgreen',alpha = 0.05)
#    #plot_covariance_ellipse((ek.xt[0,0], ek.xt[1,0]), ek.P[0:2, 0:2],std=1, facecolor='lightskyblue', alpha=0.05)
#    #plot_covariance_ellipse((ek.xt[3,0], ek.xt[4,0]), ek.P[3:5, 3:5],std=1, facecolor='lightcoral', alpha=0.05)
#    #plot_covariance_ellipse((ek.xt[5,0], ek.xt[6,0]), ek.P[5:7, 5:7],std=1, facecolor='lightgreen', alpha=0.05)
#    axRobotPosition.scatter(x[i].item(0), x[i].item(1), s=20, c = 'b')
#    t1 = axRobotPosition.quiver(x[i].item(0), x[i].item(1),np.cos(x[i].item(2)), np.sin(x[i].item(2)),linewidths=0.01, edgecolors='k')
#    axRobotPosition.plot(x[i].item(3), x[i].item(4), 'rx')
#    #plotEllipse(axRobotPosition,ek.xt.item(3), ek.xt.item(4),ek.P)
#    axRobotPosition.plot(x[i].item(5), x[i].item(6), 'gx')
#    #axRobotPosition.legend(('robot', 'landmark1', 'landmark2'))
#    axRobotPosition.set_title('Kalman filter with static landmarks')
#    axRobotPosition.relim()
#    axRobotPosition.autoscale_view()
#    #axRobotPosition.set_aspect('equal')
#        
#    plt.pause(0.05)
#    t1.set_visible(False)
##    t3.set_visible(False)
#    #t2.pop(0).remove()

def imscatter(x, y, image,degrees, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = ndimage.rotate(image, degrees)
    im = OffsetImage(im, zoom=zoom)
    #im = ndimage.rotate(im, 60)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    #ax.autoscale()
    return artists 
    
if __name__ == '__main__':
    xt,P,x,PP = example()
