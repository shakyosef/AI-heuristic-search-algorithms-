import copy
import numpy as np
import random
import math
class Agent_indexc:
    #Save an agent location on the board
    def __init__(self, row, col):
        self.row=row
        self.col=col
    def set_row(self,r):
         self.row=r
    def get_row(self):
        return self.row
    def set_col(self,c):
         self.col=c
    def get_col(self):
        return self.col
    def print_agent(self):
        print( '(',self.row,self.col,')')

class Board_mode:
    #Board object
    def __init__(self, status_board, g):
        self.status_board=status_board
        self.parents=None
        self.g=g
        self.neighbors_list=[]
        self.close=False
        self.open=False
        self.f=0
        self.heuristic=0
        self.agent_list=[]
        self.indexp=None
        self.indexf = None
        self.probability=-10000000
        self.p=0
        self.score=0
        self.motasain=False
        self.mother=None
        self.father=None
        for i in range(6):
            for j in range(6):
                if  self.status_board[i, j] == 2:
                    self.agent_list.append(Agent_indexc(i,j))
    def set_mother(self, f):
        self.mother = f
    # Saving 2 parents for genetic
    def set_father(self, f):
        self.father = f
    def get_mother(self):
        return self.mother
    def get_father(self):
        return self.father
    def set_motasain(self):
        self.motasain=True
    def get_motasain(self):
       return  self.motasain
    def get_num_agent_list(self):
        mona=0
        for i in range(6):
            for j in range(6):
                if self.status_board[i, j] == 2:
                    mona=mona+1
        return mona
    # Mutation in the board for genetic
    def to_do_motasain(self,agen_befor,row_new,col_new,dady_befor):
       for i in self.agent_list:
           if i.get_col()==agen_befor.get_col() and i.get_row()==agen_befor.get_row():
              if row_new!=6:
                row_befor= i.get_row()
                col_befor= i.get_col()
                i.set_col(col_new)
                i.set_row(row_new)
                self.status_board[row_befor, col_befor] = 0
                self.status_board[row_new, col_new] = 2
                self.motasain = True
                return self.motasain
              else:
                  row_befor = i.get_row()
                  col_befor = i.get_col()
                  self.agent_list.remove(i)
                  self.status_board[row_befor, col_befor] = 0
                  self.motasain = True
                  return self.motasain
       return self.motasain
    def get_dady_motasain(self):
        return  self.dady_motasain
    def set_p(self,p1):
        self.p=p1
    def get_p(self):
        return self.p
    def set_score(self,s):
        self.score=s
    def get_score(self):
        return self.score
    def get_agent_list(self):
      return self.agent_list
    def set_probability(self,p):
        self.probability=p
    def get_probability(self):
        return self.probability
    def set_indexs(self,p,f):
        self.indexp=p
        self.indexf=f
    def get_indexp(self):
      return self.indexp
    def get_indexf(self):
     return self.indexf
    def set_heuristic(self,h):
        self.heuristic=h
    def get_heuristic(self):
        return self.heuristic
    def get_status_board(self):
        return self.status_board
    def set_f(self,ff):
        self.f=ff
    def get_f(self):
        return self.f
    def Print_board(self):
        #  הדפסת הלוח
        for i in range(6):
            mylist = ' '
            if i == 0:
                print('      1  2  3  4  5  6 ')
            for j in range(6):
                t = ''
                if self.get_status_board()[i, j] == 1:
                    t = ' @ '
                if self.get_status_board()[i, j] == 2:
                    t = ' * '
                if self.get_status_board()[i, j] == 0:
                    t = '   '
                mylist = mylist + t
            print(i + 1, ':', mylist)
    def get_parents(self):
        return self.parents
    def set_parents(self,p):
        self.parents=p
    def get_g(self):
        return self.g
    def get_neighbors_list(self,gal_boarod):
        #Children of walking - steps are possible
        number_status = self.get_number_of_agents()
        number_gal = gal_boarod.get_number_of_agents()
        halpe_board = np.copy(self.get_status_board())
        if len(self.neighbors_list)==0:
          for i in range(6):
            for j in range(6):
                if self.get_status_board()[i, j] == 2:
                    if i != 0:
                        if self.status_board[i - 1, j] == 0:
                            halpe_board[i - 1, j] = 2
                            halpe_board[i, j] = 0
                            new_board=Board_mode(halpe_board,self.get_g()+1)
                            new_board.set_parents(self)
                            new_board.set_indexs(Agent_indexc(i,j),Agent_indexc(i - 1,j))
                            self.neighbors_list.append(new_board)
                            halpe_board = np.copy(self.get_status_board())
                    if i != 5:
                        if self.get_status_board()[i + 1, j] == 0:
                            halpe_board[i + 1, j] = 2
                            halpe_board[i, j] = 0
                            new_board = Board_mode(halpe_board, self.get_g()+1)
                            new_board.set_parents(self)
                            new_board.set_indexs(Agent_indexc(i, j), Agent_indexc(i + 1, j))
                            self.neighbors_list.append(new_board)
                            halpe_board = np.copy(self.get_status_board())
                    if j != 0:
                        if self.get_status_board()[i, j - 1] == 0:
                            halpe_board[i, j - 1] = 2
                            halpe_board[i, j] = 0
                            new_board = Board_mode(halpe_board, self.get_g()+1)
                            new_board.set_parents(self)
                            new_board.set_indexs(Agent_indexc(i, j), Agent_indexc(i, j - 1))
                            self.neighbors_list.append(new_board)
                            halpe_board = np.copy(self.get_status_board())
                    if j != 5:
                        if self.get_status_board()[i, j + 1] == 0:
                            halpe_board[i, j + 1] = 2
                            halpe_board[i, j] = 0
                            new_board = Board_mode(halpe_board, self.get_g() + 1)
                            new_board.set_parents(self)
                            new_board.set_indexs(Agent_indexc(i, j), Agent_indexc(i, j + 1))
                            self.neighbors_list.append(new_board)
                            halpe_board = np.copy(self.get_status_board())
                    if i==5 and number_status>number_gal:
                        halpe_board[i, j] = 0
                        new_board = Board_mode(halpe_board, self.get_g()+1)
                        new_board.set_parents(self)
                        new_board.set_indexs(Agent_indexc(i, j), Agent_indexc(i+1, j ))
                        self.neighbors_list.append(new_board)
                        halpe_board = np.copy(self.get_status_board())

                halpe_board = np.copy(self.status_board)

        return self.neighbors_list

    def giv_neighbors_list(self):
        return self.neighbors_list
    def set_close(self,flog):
        self.close=flog
    def get_close(self):
        return self.close
    def set_open(self,flog):
        self.open=flog
    def get_open(self):
        return self.open
    def iseqoal(self,board):
        for i in self.get_agent_list():
            flog = False
            for j in board.get_agent_list():
                if i.get_col()==j.get_col() and i.get_row()==j.get_row():
                    flog=True
            if flog==False:
               return False
        return flog
    def get_number_of_agents(self):
        mona=0
        for i in range(6):
            for j in range(6):
                if self.get_status_board()[i, j] == 2:
                    mona=mona+1
        return mona
    def get_number_of_block(self):
        mona = 0
        for i in range(6):
            for j in range(6):
                if self.get_status_board()[i, j] == 1:
                    mona = mona + 1
        return mona
    def legal012_board(self):
         for i in range(6):
            for j in range(6):
                if self.get_status_board()[i,j]!=0 and  self.get_status_board()[i,j]!=1 and  self.get_status_board()[i,j]!=2:
                    print("the board should be only with 0,1,2")
                    return False
         return True
    #Checking the integrity of the board
    def legal_board(self,starting_board):
         if self.get_number_of_block()== starting_board.get_number_of_block():
           for i in range(6):
               for j in range(6):
                   if self.get_status_board()[i,j]==1 and starting_board.get_status_board()[i,j] !=1:
                       print("the block are not in the same location, there is no way to reach the destination board ")
                       return False
                   if self.get_status_board()[i,j]!=1 and starting_board.get_status_board()[i,j] ==1:
                       print("the block are not in the same location, there is no way to reach the destination board ")
                       return False
           return True
         else:
            print("there are not the same number of block, there is no way to reach the destination board ")
            return False

    def Heuristic(self,gal_boarod):
        #Erostics - distance from Manhattan
         number_status =self.get_number_of_agents()
         number_gal=gal_boarod.get_number_of_agents()
         Heuristic_gal=np.copy(gal_boarod.get_status_board())
         Heuristic_status=np.copy(self.get_status_board())
         min=100000
         numofheuristic=0
         indexi=-1;
         indexj=-1;
         for i in range(6):
             for j in range(6):
                 if  Heuristic_gal[i,j]==2:
                     indexi = -1;
                     indexj = -1;
                     min = 100000
                     for r in range(6):
                         for c in range(6):
                             if  Heuristic_status[r, c] == 2:
                                  x = i - r
                                  x = abs(x)
                                  y=j-c
                                  y=abs(y)
                                  z=x+y
                                  if z<min:
                                      min=z
                                      indexi=r
                                      indexj=c
                     if indexi!=-1 and indexj!=-1:
                      numofheuristic=numofheuristic+min
                      Heuristic_gal[i,j]==0
                      Heuristic_status[indexi,indexj]=0


         if  number_status > number_gal:
             x=  number_status-number_gal
             min=100000
             while x>0:
                 min = 100000
                 for i in range(6):
                     for j in range(6):
                         if Heuristic_status[i,j]==2:
                             y=  abs(i-5)+1
                             if y<min:
                                 min=y
                 x=x-1
                 numofheuristic = numofheuristic + min
         return  numofheuristic

def aStarAlgo(  starting_board,goal_board,detail_output):
    open_list=[]
    open_list.append(starting_board)
    starting_board.set_heuristic(starting_board.Heuristic(goal_board))
    starting_board.set_f(starting_board.get_heuristic()+starting_board.get_g())
    close_list=[]
    while len(open_list)>0:
        min=None
        for v in open_list:
            if min==None:
                min=v
            if v.get_f()<min.get_f():
                min=v
        open_list.remove(min)
        close_list.append(min)
        if min.iseqoal(goal_board)==True:
           print_list=[]
           print_list.append(min)
           p=min.get_parents()
           while p!=None:
               print_list.append(p)
               p=p.get_parents()
           print_list.reverse()
           mona=0
           for i in  print_list:
             if mona==0:
                   print('Board 1 (starting position):')
             else:
                 if mona== len(print_list)-1:
                     print('board', mona+1, '(goal position):')
                 else:
                  print('board', mona+1,':')
             i.Print_board()
             if mona==1 and detail_output==True:
                   print('Heuristic: ' , i.get_heuristic() )
             mona=mona+1
             print('_____')
           return print_list

        else:

            for boy in min.get_neighbors_list(goal_board):
                boy.set_heuristic( boy.Heuristic(goal_board))
                boy.set_f(boy.get_heuristic() + boy.get_g())
                for o in open_list:
                     if o.iseqoal(boy)==True:
                        if boy.get_g() < o.get_g():
                            open_list.remove(o)
            for boy in min.get_neighbors_list(goal_board):
                flog=False
                y=None
                for c in close_list:
                         if c.iseqoal(boy)==True:
                            flog=True
                            y=c
                if flog==False:
                  open_list.append(boy)
                else:
                      if boy.get_g()<c.get_g():
                        close_list.remove(y)
                        open_list.append(boy)
        if min==None:
            print('path does not exist!')
            return None
    print('path does not exist!')
    return None


def hill_climbing(starting_board,goal_board):
    i=0
    min=None
    havemin=0
    starting_board.set_heuristic(goal_board)
    list_boyss=starting_board.get_neighbors_list(goal_board)
    if len(list_boyss)==0:
        print('path does not exist!')
        return None
    for boy in list_boyss:
        boy.set_heuristic(boy.Heuristic(goal_board))
        if min==None:
          min=boy
        if min.get_heuristic() > boy.get_heuristic():
            min = boy
    # Minimum son selection
    min.set_close(True)
    havemin=havemin+1
    # Boot will be performed only 5 times
    while i<6:
      if min!=None:
        if min.iseqoal(goal_board) == True:
            print_list = []
            print_list.append(min)
            p = min.get_parents()
            while p != None:
                print_list.append(p)
                p = p.get_parents()
            print_list.reverse()
            mona = 0
            for i in print_list:
                if mona == 0:
                    print('Board 1 (starting position):')
                else:
                    if mona == len(print_list) - 1:
                        print('board', mona + 1, '(goal position):')
                    else:
                        print('board', mona + 1, ':')
                i.Print_board()
                mona = mona + 1
                print('_____')
            return print_list
        else:
         flog=False
         list_boyss = min.get_neighbors_list(goal_board)
         for boy in list_boyss:
             min.set_heuristic(min.Heuristic(goal_board))
             boy.set_heuristic(boy.Heuristic(goal_board))
             if min.get_heuristic() >= boy.get_heuristic() :
                 min = boy
                 flog=True
         if flog==False:
             if min.iseqoal(goal_board) ==False:
                i = i + 1
                flog2 = False
                # Choosing a new minimal son from the starting board - a test that we have not chosen this son before
                while  flog2==False and havemin < len(starting_board.get_neighbors_list(goal_board)):
                  r = random.choice(starting_board.get_neighbors_list(goal_board))
                  if r.get_close()==False:
                    min=r
                    flog2 =True
                    min.set_close(True)
                    havemin = havemin + 1
    print('path does not exist!')
    return None

def T(t):
    start_temp = 1
    alpha = 0.01
    return start_temp- alpha*t
#where t is the iteration number and alpha is the decay
#parameter. These strategies do not guarantee
#convergence towards the global optimum, but they
#converge more rapidly towards a strong minimum
#(strong minima are solutions that lie in the near
#vicinity of the global minimum).
def simulated_annealing(starting_board,goal_board,detail_output):
    t=0
    #Sending to the temperature function
    start_temp = T(t)
    final_temp=0
    min =starting_board
    print_true=[]
    while  start_temp>final_temp:
        if min != None:
                if min.iseqoal(goal_board) == True:
                    print_list = []
                    print_list.append(min)
                    p = min.get_parents()
                    while p != None:
                        print_list.append(p)
                        p = p.get_parents()
                    print_list.reverse()
                    mona = 0
                    for i in print_list:
                        if mona == 0:
                            print('Board 1 (starting position):')
                        else:
                            if mona == len(print_list) - 1:
                                print('board', mona + 1, '(goal position):')

                            else:
                                print('board', mona + 1, ':')
                        i.Print_board()
                        if mona==0 and  detail_output==True:
                            if len(print_true)>0:
                               for boy in print_true:
                                   indexp=boy.get_indexp()
                                   indexf = boy.get_indexf()
                                   if boy.get_probability()!=-10000000 :
                                        print('action:','(',indexp.get_row()+1,',',indexp.get_col()+1,')','-->' ,'(',indexf.get_row()+1,',',indexf.get_col()+1,')','probability:',boy.get_probability())
                        mona = mona + 1
                        print('_____')
                    return print_list
                else:
                 numclose=0
                 havemin =False
                 list_neighbors=min.get_neighbors_list(goal_board)
                 if len( list_neighbors) == 0:
                     print('path does not exist!')
                     return None
                 # Finding a son - a test that we did not choose the same son
                 while havemin==False:
                   if numclose==len(list_neighbors)-1:
                       print('path does not exist!')
                       return None
                   else:
                    neighbor = random.choice(list_neighbors)
                    if  neighbor.get_close()==False:
                       neighbor.set_close(True)
                       numclose=numclose+1
                       # For printing
                       if t==0:
                        print_true.append(neighbor)
                       min.set_heuristic(min.Heuristic(goal_board))
                       neighbor.set_heuristic(neighbor.Heuristic(goal_board))
                       cost_diff =  min.get_heuristic()- neighbor.get_heuristic()
                       # Check if possible in their son positive - if so choose it if not a choice if the probability is greater than a random number
                       neighbor.set_probability(math.exp(cost_diff / start_temp))
                       if  cost_diff >0:
                         min=neighbor
                         min.set_probability(1)
                         havemin = True
                       else:
                         x=random.uniform(0, 1)
                         if  x < math.exp(cost_diff / start_temp):
                             min = neighbor
                             havemin = True
                         else:
                            havemin = False
                 if min.iseqoal(goal_board) == True:
                     t=t
                 else:
                     #Promoting the step - updating the temperature by the temperature function
                  t = t + 1
                  start_temp = T(t)
        else:
           print('path does not exist!')
           return None
    print('path does not exist! ')
    return None

def selection_sort(neighbors_list,goal_board):
    #Sort array from smallest to grow according to heuristics
 for boy in neighbors_list:
     boy.set_heuristic(boy.Heuristic(goal_board))
 for i in range(len(neighbors_list)):
     min_index = i
     for j in range(i, len(neighbors_list)):
        if neighbors_list[min_index].get_heuristic()>neighbors_list[j].get_heuristic():
            min_index=j
     neighbors_list[i],neighbors_list[min_index] = neighbors_list[min_index], neighbors_list[i]
def k_boys(status_board,goal_board,k):
    # Choose the 3 best boys from only 1 board
    list_open=[]
    index=0
    neighbors_list = status_board.get_neighbors_list(goal_board)
    if len( neighbors_list) == 0:
        return None
    status_board.set_heuristic( status_board.Heuristic(goal_board))
    selection_sort(neighbors_list, goal_board)
    size=len(neighbors_list)
    if  size==0:
        return None
    if size >= k:
        while index<k:
           list_open.append(neighbors_list[index])
           index=index+1
        return  list_open
    else:
        if size==1:
            list_open.append(neighbors_list[0])
            return list_open
        if size==2:
            list_open.append(neighbors_list[0])
            list_open.append(neighbors_list[1])
            return list_open

def besat_k_boys_all(list_boy,k):
    #From all the best boys of all the boards - choose the 3 best boys of all
    list_open = []
    index=0
    size = len(list_boy)
    if size==0:
        return None
    if  size>k:
      while index<k:
        list_open.append(list_boy[index])
        index=index+1
      return list_open
    if size==1:
        list_open.append(list_boy[0])
        return list_open
    if size==2:
        list_open.append(list_boy[0])
        list_open.append(list_boy[1])
        return list_open

def a_local_beam_search(starting_board,goal_board,detail_output):
    k=3
    index=0
    mona=0
    list_open=[]
    print_list = []
    better = True
    list_open=k_boys(starting_board,goal_board,k)
    if list_open==None:
        better = False
    while  better==True:
          for i in list_open:
              if i.iseqoal(goal_board) == True:
                  print_list.append(i)
                  p = i.get_parents()
                  while p != None:
                      print_list.append(p)
                      p = p.get_parents()
                  print_list.reverse()
                  mona = 0
                  for i in print_list:
                      if mona == 0:
                          print('Board 1 (starting position):')
                      else:
                          if mona == len(print_list) - 1:
                              print('board', mona + 1, '(goal position):')
                          else:
                              print('board', mona + 1, ':')
                      i.Print_board()
                      print('_____')
                      if mona==0 and detail_output == True:
                          boys = []
                          boys = i.get_neighbors_list(goal_board)
                          selection_sort(boys, goal_board)
                          if len(boys)>3:
                           print('Board 2a:')
                           boys[0].Print_board()
                           print('_____')
                           print('Board 2b:')
                           boys[1].Print_board()
                           print('_____')
                           print('Board 2c:')
                           boys[2].Print_board()
                           print('_____')
                      mona = mona + 1
                  return print_list
          list_boy = []
          # From each board take 3 best boys
          for i in list_open:
              list_hlpe=k_boys(i, goal_board, k)
              if  list_hlpe!= None:
                  for j in list_hlpe:
                      list_boy.append(j)

          list_open=[]
          # Stop after 500 runs or there are no more best boys
          if  mona==500 or len(list_boy)==0:
              better = False
          else:
              # Taking 3 best boys of all
              selection_sort(list_boy, goal_board)
              list_open=besat_k_boys_all(list_boy,k)
              if list_open==None:
                  better = False
          mona=mona+1
    print('path does not exist!')
    return print_list


# Sorting from the smallest to growing by heuristics
def selection_sort_h(neighbors_list,goal_board):
 for boy in neighbors_list:
     boy.set_heuristic(boy.Heuristic(goal_board))
 for i in range(len(neighbors_list)):
     min_index = i
     for j in range(i, len(neighbors_list)):
        if neighbors_list[min_index].get_heuristic()>neighbors_list[j].get_heuristic():
            min_index=j
     neighbors_list[i],neighbors_list[min_index] = neighbors_list[min_index], neighbors_list[i]


# In order for the lowest yursica to receive the highest score, we will take the maximum heuristics and from it we will remove the heuristics of each board and another 1 and this is the score
def get_score(goal_board,boys):
    #Scoreboard - for the genetic path
    # If it is not possible to create a population of ten creators the sons of the boys
    max=-88888888
    population = []
    if len(boys) == 0:
        return None
    else:
        selection_sort_h(boys, goal_board)
    if len(boys) > 10:
        for i in range(10):
            population.append(boys[i])
    else:
        for i in  boys:
            population.append(i)
    while len( population) <10:
           boys2=[]
           x= 10-len( population)
           for i in population:
               boys2= boys2+i.get_neighbors_list(goal_board)
           selection_sort_h(boys, goal_board)
           if len(boys2) > x:
               for i in range(x):
                   population.append(boys2[i])
           else:
               for i in boys2:
                   population.append(i)
    # Finding a maximum in the list
    for i in population:
        if max==-88888888:
            max=i.get_heuristic()
        if i.get_heuristic()>max:
            max=i.get_heuristic()
    for i in population:
        # Add a constant to reduce significant gaps
        x=max-i.get_heuristic()+2
        i.set_score(x)
    return  population

def get_p_list(goal_board,boys):
    # Probability by score - for the genetic path
    population = []
    score = 0
    population=get_score(goal_board,boys)
    if population==None:
        return None
    for i in population:
          score = score + i.get_score()
    for i in population:
          i.set_p(i.get_score() / score)
    selection_sort_p(population)
    return  population
# We have circumvented the problem to the maximum- now the maximum probability will be for the one with the lowest heuristics
def selection_sort_p(boys):
    # Sorting from large to small according to the probability of selection
 for i in range(len(boys)):
     min_index = i
     for j in range(i, len(boys)):
        if boys[min_index].get_p()>boys[j].get_p():
            min_index=j
     boys[i],boys[min_index] = boys[min_index],boys[i]
 boys.reverse()

def get_to_random(max):
    # Select 2 random numbers - different
    # For genetic- for selecting 2 parents in a particular area of probability
    randon1 = random.uniform(0, 1)
    randon2 = random.uniform(0, 1)
    while  randon1== randon2:
        randon1 = random.uniform(0, 1)
        randon2 = random.uniform(0, 1)
    return  randon1,randon2
def get_parens(population):
    #Finding 2 parents - for genetic
    #Receiving 2 parents - Lottery of values 1 and finding a father Lottery of values 2 and finding a mother
    father = None
    mother = None
    max=0
    for i in population:
        if i.get_p()>max:
            max=i.get_p()
    population_copy=copy.copy(population)
    randon1, randon2  = get_to_random(max)
    randon3, randon4 =   get_to_random(max)
    min_random1 = randon1
    max_random1 = randon3
    if min_random1 > randon3:
       min_random1 = randon3
       max_random1 = randon1
    min_random2 = randon2
    max_random2 = randon4
    if min_random2 > randon4:
     min_random2 = randon4
     max_random2 = randon2
    flog=False
    for i in population:
        if i.get_p() >= min_random1 and  i.get_p()<= max_random1:
            father=i
            for j in population_copy:
                if j.get_p() >= min_random2 and j.get_p()<= max_random2:
                    mother = j
                    if  mother.iseqoal(father)==False:
                        flog=True
                        return father, mother
                    else:
                        population_copy.remove(mother)
                        if len(population_copy)==0:
                           population_copy=copy.copy(population)
    return None, None
#
def new_father(row_new,col_new,row_old,col_old,boy):
    #For genetic
    #Defining a father-son created by mating- to prevent a situation of performing 2 steps together
    father_for_boy=copy.copy(boy.get_status_board())
    father_for_boy[row_new,col_new]=2
    if row_old!=7 and col_old!=7:
       father_for_boy[row_old, col_old] = 0
    father_new_board = Board_mode( father_for_boy, 0)
    return  father_new_board

def set_boy( father,boy):
    #For genetic
    #After pairing check which agent has progressed more step-1 setting Dad for each board with agent progress
    agent_father = father.get_agent_list()
    agent_boy= boy.get_agent_list()
    bord_father =  father.get_status_board()
    bord_boy = boy.get_status_board()
    boy_new=copy.copy(boy)
    boy_old = boy
    flog=False
    for i in range(6):
        for j in range(6):
            if   bord_father[i,j]==2 and bord_boy[i,j]==0:
                flog = True
                if bord_boy[i,j]==0:
                    if i!=5 and i!=0:
                        if bord_boy[i-1,j]==2 and bord_father[i-1,j]==0:
                            boy_new= new_father(i,j,i-1,j,boy_old)
                            boy_old.set_parents( boy_new)
                            boy_old=boy_new
                        if  bord_boy[i+1,j]==2 and bord_father[i+1,j]==0:
                            boy_new = new_father(i, j, i + 1, j, boy_new)
                            boy_old.set_parents(boy_new)
                            boy_old = boy_new
                    if i==5 :
                        flog5=False
                        if bord_boy[i - 1, j] == 2 and bord_father[i-1,j]==0:
                            flog5=True
                            boy_new = new_father(i, j, i - 1, j, boy_new)
                            boy_old.set_parents(boy_new)
                            boy_old = boy_new
                    if i==0:
                        if bord_boy[i + 1, j] == 2 and bord_father[i+1,j]==0:
                            boy_new = new_father(i, j, i + 1, j, boy_new)
                            boy_old.set_parents(boy_new)
                            boy_old = boy_new
                    if j !=0 and j!=5:
                        if bord_boy[i, j-1] == 2 and bord_father[i,j-1]==0:
                            flog5=True
                            boy_new = new_father(i, j, i, j-1, boy_new)
                            boy_old.set_parents(boy_new)
                            boy_old = boy_new
                        if bord_boy[i , j+1] == 2 and bord_father[i,j+1]==0:
                            flog5 = True
                            boy_new = new_father(i, j, i, j + 1, boy_new)
                            boy_old.set_parents(boy_new)
                            boy_old = boy_new
                    if j==0:
                        if bord_boy[i, j + 1] == 2 and bord_father[i,j+1]==0:
                            flog5 = True
                            boy_new = new_father(i, j, i, j + 1, boy_new)
                            boy_old.set_parents(boy_new)
                            boy_old = boy_new
                    if j==5:
                        if bord_boy[i, j - 1] == 2 and bord_father[i,j-1]==0:
                            flog5 = True
                            boy_new = new_father(i, j, i, j - 1, boy_new)
                            boy_old.set_parents(boy_new)
                            boy_old = boy_new
                    if i==5 and flog5==False:
                        flog5=True
                        boy_new = new_father(i, j,7,7, boy_new)
                        boy_old.set_parents(boy_new)
                        boy_old = boy_new
            bord_boy =boy_old.get_status_board()

    boy_old.set_parents(father)

def croos (father,mother, amunt_agent_goal):
    # Parental mating - for genetic
    boy1 = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    boy2 = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    agent_father =father.get_agent_list()
    agent_mother = mother.get_agent_list()
    agent_togder=[]
    agent_max = len( agent_father)
    agent_min = amunt_agent_goal
    if len(agent_mother) > agent_max:
        agent_max = len(agent_mother)
    # Lottery number cut - by columns
    median= random.randint(0,5)
    for i in range(6):
        for j in range(6):
            x = copy.copy(father.get_status_board()[i, j])
            y = copy.copy(mother.get_status_board()[i, j])
            if x == y:
                boy1[i, j]= x
                boy2[i, j]= x
                agent_togder.append(Agent_indexc(i,j))
            else:
                if j < median:
                    boy1[i, j]= x
                    boy2[i, j]=y
                else:
                    boy1[i, j]= y
                    boy2[i, j]= x
    flogboy1=False
    flogboy2 = False
    # Creating 2 children.
    # Child 1 Part A from father Part B from mother
    # Child 2 Part A From Mother Part B From Father
    boy1_board = Board_mode(boy1, 0)
    boy2_board = Board_mode(boy2, 0)
    agent_boy1 = boy1_board.get_agent_list()
    agent_boy2= boy2_board.get_agent_list()
    amunt_agent_boy1=len(agent_boy1)
    amunt_agent_boy2 = len(agent_boy2)
    #Check if the child created is legal - if no agents have disappeared
    if amunt_agent_boy1 >= agent_min and amunt_agent_boy1<=  agent_max:
                flogboy1=True

    if amunt_agent_boy2 >= agent_min and amunt_agent_boy2 <= agent_max:
              flogboy2 = True

    if  flogboy1==False and  flogboy2==False:
        return None, None
    else:
        #If the child is legal - setting a father for the child in order to prevent more step 1
        if flogboy1==True and  flogboy2==True:
            set_boy(father, boy1_board)
            set_boy(father, boy2_board)
            boy1_board.set_mother(mother)
            boy1_board.set_father(father)
            boy2_board.set_mother(mother)
            boy2_board.set_father(father)
            return boy1_board,boy2_board
        if  flogboy1==True and  flogboy2==False:
            set_boy(father, boy1_board)
            boy1_board.set_mother(mother)
            boy1_board.set_father(father)
            return boy1_board, None
        if  flogboy2==True and  flogboy1==False:
            set_boy(father, boy2_board)
            boy2_board.set_mother(mother)
            boy2_board.set_father(father)
            return  None,boy2_board


def insamplas( boys_list,population):
    #Convergence test - for genetic
    #Check if all my boards in a certain generation and the next generation remain in the same score - that is, there is no progress
    flog=False
    same=0
    for i in population:
        for j in boys_list:
            if i.get_score()==j.get_score()-1 or  i.get_score()==j.get_score()+1 or  i.get_score()==j.get_score():
                same = same + 1
    if same>=len(population):
        flog=True
    return flog


def do_motasain(boys,amunt_agent_goal):
    #Mutation for genetic
  boy_motasain = boys
  dady_motasain=copy.copy(boys)
  if  boy_motasain.get_motasain()==False:
    list_agent=boy_motasain.get_agent_list()
    amunt_agent_boy=len(list_agent)
    cantoout=False
    #Check if there is a need for an agent to leave the board
    if amunt_agent_boy>amunt_agent_goal:
        cantoout=True
        # Lottery of a specific agent from all the agents on the board
    agen_motasain = random.choice(list_agent)
    list_col=[]
    list_row = []
    list_chois =[]
    # Include in the list all the rows that the agent can advance on and all the columns that the agent can advance on
    for i in range(6):
        for j in range(6):
            if boy_motasain.get_status_board()[i,j]==0:
              if agen_motasain.get_row()==i:
                 if agen_motasain.get_col()-1==j:
                     list_col.append(j)
                 if  agen_motasain.get_col()+1==j:
                     list_col.append(j)
              if agen_motasain.get_col()==j:
                  if agen_motasain.get_row()-1==i:
                      list_row.append(i)
                  if agen_motasain.get_row()+1==i:
                      list_row.append(i)
              if agen_motasain.get_row()==5 and cantoout==True:
                  list_row.append(6)
    if len( list_col)!=0 and len(list_row)!=0:
        list_chois.append('c')
        list_chois.append('r')
        # Lottery Whether to select a row or column step
        # Lottery where to proceed and execution of the mutation
        chois=random.choice(list_chois)
        if chois.find('c'):
            row_new=agen_motasain.get_row()
            new_col=random.choice(list_col)
            return boy_motasain.to_do_motasain(  agen_motasain, row_new, new_col,dady_motasain)
        if chois.find('r'):
            new_col=agen_motasain.get_col()
            row_new=random.choice(list_row)
            return  boy_motasain.to_do_motasain(agen_motasain, row_new, new_col,dady_motasain)
    if  len( list_col)!=0:
        row_new = agen_motasain.get_row()
        new_col = random.choice(list_col)
        return  boy_motasain.to_do_motasain(agen_motasain, row_new, new_col,dady_motasain)
    if len(list_row)!=0:
        new_col = agen_motasain.get_col()
        row_new = random.choice(list_row)
        return boy_motasain.to_do_motasain(agen_motasain, row_new, new_col,dady_motasain)


def  genetic(starting, goal, detail_output):
 agent_goal=goal.get_agent_list()
 amunt_agent_goal=len(agent_goal)
 epsilon=0.2
 boys_list= starting.get_neighbors_list(goal)
 population=[]
 same_plase=0
 if len( boys_list)==0:
     print('path does not exist!')
     return None
 # Probabilities for the population
 population=get_p_list(goal, boys_list)
 for i in population:
     i.set_parents(starting)
 boys_list=[]
 mona=0
 end=False
 # Run until a convergence has occurred 100 times or a solution is found
 while same_plase != 100 and end==False:
    for i in population:
        if i.iseqoal(goal) == True:
            print_list=[]
            print_list.append(i)
            dady = i.get_parents()
            dady_befor=None
            while dady!=None :
                if dady_befor==None or dady.iseqoal(dady_befor)==False:
                    dady_befor=dady
                    print_list.append( dady)
                dady = dady.get_parents()
            print_list.reverse()
            mona = 0
            for i in print_list:
                if mona == 0:
                    print('Board 1 (starting position):')
                else:
                    if mona == len(print_list) -1:
                        print('board', mona + 1, '(goal position):')
                    else:
                        print('board', mona + 1, ':')
                i.Print_board()
                print('_____')
                mona=mona+1
            end=True
    father = None
    mother = None
    #Selection of 2 parents
    father, mother = get_parens(population)
    if father != None and mother != None:
      boy1=None
      boy2=None
      while boy1==None or boy2==None:
          # Creating 2 children from the parents
         boy1, boy2 = croos(father, mother, amunt_agent_goal)
      if boy1!=None and boy2!=None:
         boys_list.append(boy1)
         boys_list.append(boy2)
      if boy1 != None and boy2==None:
          boys_list.append(boy1)
      if boy2!=None and  boy1 == None:
          boys_list.append(boy2)
    if len(boys_list) == 10:
               mona = mona + 1
               for i in boys_list:
                 randon = random.random()
                 append = False
                 # Test whether to make a mutation
                 if randon <= epsilon:
                   append = do_motasain(i, amunt_agent_goal)
               boys = get_score(goal, boys_list)
               flog = insamplas(boys_list, population)
               if flog == True:
                   same_plase = same_plase + 1
               population = []
               # Providing probabilities to the new population
               population = get_p_list(goal, boys_list)
               boys_list = []
 if detail_output==True:
   boy=population[0]
   motasain= boy.get_motasain()
   father=  boy.get_father()
   mother=  boy.get_mother()
   if motasain!=None and  father!=None and  mother!=None:
    p_father = father.get_p()
    p_mother =  mother.get_p()
    print('Starting board 1 (probability of selection from population::<', p_father,'>):')
    father.Print_board()
    print('-----')
    print('Starting board 2 (probability of selection from population::<',p_mother,'>):')
    mother.Print_board()
    print('-----')
    if motasain==True:
      print('Result board (mutation happened::<yes>):')
    else:
       print('Result board (mutation happened::<no>):')
    boy.Print_board()
 if end==False:
  print('path does not exist!')
 return None


def find_path(starting_board,goal_board,search_method,detail_output):
      starting = Board_mode(starting_board, 0)
      goal = Board_mode(goal_board, 0)
      slegsl=starting.legal012_board()
      glegsl= goal.legal012_board()
      if slegsl==True and glegsl==True:
        if  starting.legal_board(goal):
            if search_method==1:
                aStarAlgo(starting,goal,detail_output)
            if search_method==2:
                hill_climbing( starting, goal)
            if search_method==3:
                simulated_annealing(starting,goal,detail_output)
            if search_method==4:
                a_local_beam_search(starting,goal,detail_output)
            if search_method == 5:
                genetic(starting, goal, detail_output)
























