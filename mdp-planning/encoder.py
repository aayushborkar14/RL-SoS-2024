import argparse

parser = argparse.ArgumentParser()

def main():
    # p1=(0,0)
    # p2=(2,2)
    # o1=(3,3)
    parser.add_argument("--opponent", type=str)
    parser.add_argument("--p", type=float)
    parser.add_argument("--q", type=float)
    args = parser.parse_args()
    Encoder(args.opponent,args.p,args.q)

         
    
class Encoder():
    def __init__(self,opponent_file,p,q):    
        self.p=p
        self.q=q
        self.mapper={}
        self.states=[None]*8194
        curr_state=2
        self.mapper[0]=0
        with open(opponent_file) as file:
            for line in file:
                line2 = line.rstrip()
                words = line2.split()
                if words[0]=="state":
                    continue
                state=words[0]
                self.mapper[state]=curr_state
                self.states[curr_state]=[state,float(words[1]),float(words[2]),float(words[3]),float(words[4])]
                curr_state+=1
        print("numStates 8194")
        print("numActions 10")
        print("end 0 1")
        self.transitions_printer()
        print("mdptype episodic")
        print("discount 1.00")
    
    def transitions_printer(self):
        lx=[0,0,-1,1]
        ly=[-1,1,0,0]
        for x in range(2,8194):
            v=self.states[x]
            p1=return_point(v[0][0:2])
            p2=return_point(v[0][2:4])
            o=return_point(v[0][4:6])
            possession=int(v[0][6])
            for i in range(0,4):
                if v[i+1]>0:
                    o_new=(o[0]+lx[i],o[1]+ly[i])
                    self.action_0(x,p1,p2,o,o_new,v[i+1],possession)
                    self.action_1(x,p1,p2,o,o_new,v[i+1],possession)
                    self.action_2(x,p1,p2,o,o_new,v[i+1],possession)
                    self.action_3(x,p1,p2,o,o_new,v[i+1],possession)
                    self.action_4(x,p1,p2,o,o_new,v[i+1],possession)
                    self.action_5(x,p1,p2,o,o_new,v[i+1],possession)
                    self.action_6(x,p1,p2,o,o_new,v[i+1],possession)
                    self.action_7(x,p1,p2,o,o_new,v[i+1],possession)
                    self.action_8(x,p1,p2,o,o_new,v[i+1],possession)
                    self.action_9(x,p1,p2,o,o_new,v[i+1],possession)
       
       
  #ACTION FUNCTIONS          
    def action_0(self,curr_state,p1,p2,o,o_new,mul,possession):
        p1_new=(p1[0],p1[1]-1)
        if check_position(p1_new)==False:
            return
        prob=mul
        if possession==2:
            prob*=(1-self.p)
        else:
            prob*=(1-(2*self.p))
        new_state=return_string(p1_new)+return_string(p2)+return_string(o_new)+str(possession)

        if possession==2:
            print("transition {} 0 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
            return
        if p1_new==o_new:
            prob*=0.5
        elif p1_new==o and o_new==p1:
            prob*=0.5
        print("transition {} 0 {} 0 {}".format(curr_state,self.mapper[new_state],prob))

    def action_1(self,curr_state,p1,p2,o,o_new,mul,possession):
        p1_new=(p1[0],p1[1]+1)
        if check_position(p1_new)==False:
            return
        prob=mul
        if possession==2:
            prob*=(1-self.p)
        else:
            prob*=(1-(2*self.p))
        new_state=return_string(p1_new)+return_string(p2)+return_string(o_new)+str(possession)

        if possession==2:
            print("transition {} 1 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
            return
        if p1_new==o_new:
            prob*=0.5
        elif p1_new==o and o_new==p1:
            prob*=0.5
        print("transition {} 1 {} 0 {}".format(curr_state,self.mapper[new_state],prob))

    def action_2(self,curr_state,p1,p2,o,o_new,mul,possession):
        p1_new=(p1[0]-1,p1[1])
        if check_position(p1_new)==False:
            return
        prob=mul
        if possession==2:
            prob*=(1-self.p)
        else:
            prob*=(1-(2*self.p))
        new_state=return_string(p1_new)+return_string(p2)+return_string(o_new)+str(possession)

        if possession==2:
            print("transition {} 2 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
            return
        if p1_new==o_new:
            prob*=0.5
        elif p1_new==o and o_new==p1:
            prob*=0.5
        print("transition {} 2 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
        
    def action_3(self,curr_state,p1,p2,o,o_new,mul,possession):
        p1_new=(p1[0]+1,p1[1])
        if check_position(p1_new)==False:
            return
        prob=mul
        if possession==2:
            prob*=(1-self.p)
        else:
            prob*=(1-(2*self.p))
        new_state=return_string(p1_new)+return_string(p2)+return_string(o_new)+str(possession)

        if possession==2:
            print("transition {} 3 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
            return
        if p1_new==o_new:
            prob*=0.5
        elif p1_new==o and o_new==p1:
            prob*=0.5
        print("transition {} 3 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
        
    def action_4(self,curr_state,p1,p2,o,o_new,mul,possession):
        p2_new=(p2[0],p2[1]-1)
        if check_position(p2_new)==False:
            return
        prob=mul
        if possession==1:
            prob*=(1-self.p)
        else:
            prob*=(1-(2*self.p))
        new_state=return_string(p1)+return_string(p2_new)+return_string(o_new)+str(possession)

        if possession==1:
            print("transition {} 4 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
            return
        if p2_new==o_new:
            prob*=0.5
        elif p2_new==o and o_new==p2:
            prob*=0.5
        print("transition {} 4 {} 0 {}".format(curr_state,self.mapper[new_state],prob))

    def action_5(self,curr_state,p1,p2,o,o_new,mul,possession):
        p2_new=(p2[0],p2[1]+1)
        if check_position(p2_new)==False:
            return
        prob=mul
        if possession==1:
            prob*=(1-self.p)
        else:
            prob*=(1-(2*self.p))
        new_state=return_string(p1)+return_string(p2_new)+return_string(o_new)+str(possession)

        if possession==1:
            print("transition {} 5 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
            return
        if p2_new==o_new:
            prob*=0.5
        elif p2_new==o and o_new==p2:
            prob*=0.5
        print("transition {} 5 {} 0 {}".format(curr_state,self.mapper[new_state],prob))

    def action_6(self,curr_state,p1,p2,o,o_new,mul,possession):
        p2_new=(p2[0]-1,p2[1])
        if check_position(p2_new)==False:
            return
        prob=mul
        if possession==1:
            prob*=(1-self.p)
        else:
            prob*=(1-(2*self.p))
        new_state=return_string(p1)+return_string(p2_new)+return_string(o_new)+str(possession)

        if possession==1:
            print("transition {} 6 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
            return
        if p2_new==o_new:
            prob*=0.5
        elif p2_new==o and o_new==p2:
            prob*=0.5
        print("transition {} 6 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
        
    def action_7(self,curr_state,p1,p2,o,o_new,mul,possession):
        p2_new=(p2[0]+1,p2[1])
        if check_position(p2_new)==False:
            return
        prob=mul
        if possession==1:
            prob*=(1-self.p)
        else:
            prob*=(1-(2*self.p))
        new_state=return_string(p1)+return_string(p2_new)+return_string(o_new)+str(possession)

        if possession==1:
            print("transition {} 7 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
            return
        if p2_new==o_new:
            prob*=0.5
        elif p2_new==o and o_new==p2:
            prob*=0.5
        print("transition {} 7 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
        
    def action_8(self,curr_state,p1,p2,o,o_new,mul,possession):
        prob=mul
        prob*=(self.q-(0.1*max(abs(p1[0]-p2[0]),abs(p1[1]-p2[1]))))
        if colinearity(p1,p2,o_new):
            prob*=0.5
        if possession==1:
            new_state=return_string(p1)+return_string(p2)+return_string(o_new)+str(2)
            print("transition {} 8 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
        if possession==2:
            new_state=return_string(p1)+return_string(p2)+return_string(o_new)+str(1)
            print("transition {} 8 {} 0 {}".format(curr_state,self.mapper[new_state],prob))
            
    
    
    def action_9(self,curr_state,p1,p2,o,o_new,mul,possession):
        prob=mul
        if possession==1:
            prob*=(self.q-(0.2*(3-p1[1])))
        if possession==2:
            prob*=(self.q-(0.2*(3-p2[1])))
        if o_new==(1,3) or o_new==(2,3):
            prob*=0.5
        print("transition {} 9 {} 1 {}".format(curr_state,0,prob))
            




#Helper functions
def check_position(p1):
    return p1[0]>=0 and p1[1]>=0 and p1[0]<4 and p1[1]<4
def return_string(p1):
    p1=((4*p1[0])+(p1[1]))
    p1+=1
    p1=str(p1)
    if(len(p1)==1):
        p1="0"+p1
    return p1
        

def colinearity(p1,p2,o1):
    if p1==p2==o1:
        return True
    x1=p1[0]
    x2=p2[0]
    x=o1[0]
    y1=p1[1]
    y2=p2[1]
    y=o1[1]
    minx=min(x1,x2)
    maxx=max(x1,x2)
    maxy=max(y1,y2)
    miny=min(y1,y2)
    if p1==o1:
        m1=(y2-y)/(x2-x) if x2-x!=0 else None
    elif p2==o1:
        m1=(y1-y)/(x1-x) if x1-x!=0 else None
    else:
        m1=(y2-y)/(x2-x) if x2-x!=0 else None
    
    m2=((y2-y1)/(x2-x1)) if x2-x1!=0 else None
    if m1!=m2:
        return False
    if m2!=1 and m2!=0 and m2!=-1 and m2!=None:
        return False
    if minx<=x<=maxx and miny<=y<=maxy:
        return True
    return False
    # if (x1==x2):
    #     if x!=x2:
    #         return False
    #     return (((y-y1)*(y-y2))<=0)
    # if x==x2:
    #     return False
    # m=(y2-y1)/(x2-x1)
    # if m!=1 and m!=-1 and m!=0:
    #     return False
    # m1=(y2-y)/(x2-x)
    # if m!=m1:
    #     return False
    # return (((y-y1)*(y-y2))<=0)

    
            
def return_point(p1):
    p1=int(p1)
    p1-=1
    row=p1//4
    column=p1%4
    return (row,column)
    



if __name__ == "__main__":
    main()