from collections import Counter
import math
import random
import sys
import os.path

def read_datafile(fname, attribute_data_type = 'integer'):
   inf = open(fname,'r')
   lines = inf.readlines()
   inf.close()
   #--
   X = []
   Y = []
   for l in lines:
      ss=l.strip().split(',')
      temp = []
      for s in ss:
         if attribute_data_type == 'integer':
            temp.append(int(s))
         elif attribute_data_type == 'string':
            temp.append(s)
         else:
            print("Unknown data type");
            exit();
      X.append(temp[:-1]) #X=Params, list of lists
      Y.append(int(temp[-1])) #Y=Results
   return X, Y

#===
class DecisionTree:
   def __init__(self, split_random, depth_limit, curr_depth = 0, default_label = 1):
      self.split_random = split_random # if True splits randomly, otherwise splits based on information gain 
      self.depth_limit = depth_limit
      self.gains = []
      self.usedBests = []
      self.default_label = default_label
      self.loopCount = 0
      self.gainIdxs = []

   def chooseAttribute(self, attributes, Y):
      maxGainIdx = -1
      if len(self.gains) == 0:
         for att in attributes:
            self.gains.append(self.gain(att, Y))
            if maxGainIdx is -1:
               maxGainIdx = len(self.gains)-1
            elif self.gains[-1] > self.gains[maxGainIdx]:
               maxGainIdx = len(self.gains)-1
      else:
         m = None
         for i in range(len(self.gains)):
            if i not in self.usedBests:
               if m is None:
                  m = i
               elif self.gains[m] < self.gains[i]:
                  m = i
         maxGainIdx = m
##         if maxGainIdx in self.usedBests:
##            print("SOMETHING WENT WRONG\n")



#outdated:
##         for i in range(len(self.gains)):
##            #if maxGainIdx == -1 and (not i in self.usedBests):
##            if maxGainIdx == -1:
##               maxGainIdx = i
##            elif (self.gains[i] > self.gains[maxGainIdx]) and (not i in self.usedBests):
##               maxGainIdx = i
      #when len(self.gains) == len(self.usedBets), i = -1 (using commented out if statement, otherwise it's 0)
      self.usedBests.append(maxGainIdx)
      return maxGainIdx

   def randomAttribute(self, attributes):
      indices = list(range(len(attributes)))
      self.usedBests.sort()
      for i in reversed(self.usedBests):
         del indices[i]
      choice = random.choice(indices)
      self.usedBests.append(choice)
      return choice

   def remainder(self, att, Y_train):
      probs = self.getProbs(att, Y_train)
      fAtt = att.count(0)/len(att) #chance of attribute being false (0)
      tAtt = att.count(1)/len(att) #chance of attribute being true (1)
      rem = (fAtt* self.algorithmI(probs[0][0], probs[0][1])) + (tAtt * self.algorithmI(probs[1][0], probs[1][1])) #order of algorithmI params based on gainAndRemainder image
      return rem
      

      
      #v is 2 for this data set
      #need goal to tell if a 0 or 1 is negative or positive
      
   def getProbs(self, att, Y_train): #probs = [[Pr(Y = 0 | A = 0), Pr(Y=1 | A=0)], [Pr(Y = 0 | A = 1), Pr(Y=1 | A=1)]
      if len(att) != len(Y_train):
         print("Data lengths don't match")
         return None #failure

      probs = [[0,0],[0,0]]
      for i in range(len(Y_train)):
         if att[i] == 0:
            if Y_train[i] == 0:
               probs[0][0] += 1
            elif Y_train[i] == 1:
               probs[0][1] += 1

         elif att[i] == 1:
            if Y_train[i] == 0:
               probs[1][0] += 1
            elif Y_train[i] == 1:
               
               probs[1][1] +=1
      

      fAtt = att.count(0) #total att = false
      tAtt = att.count(1) #total att = true
      if fAtt == 0:
          probs[0][0] = 0
          probs[0][1] = 0
      else:
          for i in range(len(probs[0])):
              probs[0][i] = probs[0][i]/fAtt
      if tAtt == 0:
          probs[1][0] = 0
          probs[1][1] = 0
      else:
          for i in range(len(probs[1])):
              probs[1][i] = probs[1][i]/tAtt



      return probs

   def algorithmI(self, posP, negP): #I think negP and posP are backwards -DP
       
       part1 = 0
       part2 = 0
       if posP != 0:
           part1 = posP * math.log(posP, 2)
       if negP != 0:
           part2 = negP * math.log(negP, 2)
       i = 0 - part1 - part2
       return i

   def gain(self, att, Y):
      #Gain(A) = I([Pr(positive result)], [Pr(negative result)]) - Remainder(A)
      posP = Y.count(1)/len(Y) #Pr(positive result)
      negP = Y.count(0)/len(Y) #Pr(negative result)
      i = self.algorithmI(posP, negP)
      rem = self.remainder(att, Y)
      return i - rem

   def decisionTreeLearning(self, attributes, X_train, Y_train, parent):
      #An example is described by the values of the attributes and the value of the goal predicate.
      #We call the value of the goal predicate the classification of the example.

      #attributes: list of lists, 37 lists of 2000 items
      
      if Y_train.count(1) > Y_train.count(0):
        majorityVal = 1 #True
      else:
        majorityVal = 0 #False
      self.loopCount += 1
      #print("Loop count: " + str(self.loopCount))
      if len(Y_train) == 0:
         #print("ret1")
         return majorityVal

      elif 0 not in Y_train or 1 not in Y_train:
         #print("ret2 because " + str(Y_train[0]))
         
         return Y_train[0] #"We call the value of the goal predicate the classification of the example"

      elif len(attributes) <= len(self.usedBests):
          #print("ret3")
          return majorityVal
      #else:
      #   print(str(len(attributes)) + " " + str(len(self.usedBests)))
      #   print(self.usedBests)
      if parent is not None:
         
         if parent.depth == self.depth_limit: #if parent has hit the depth limit, these 'subtrees' will just be leaves
            #print("ret4")
            return majorityVal
      best = None
      if not self.split_random:
         best = self.chooseAttribute(attributes, Y_train)
         
         #print(best)
      else:
         best = self.randomAttribute(attributes)
      tree = Tree(best, parent) #bad when limit >5
      for v in [0,1]:
         newX = [] #list of X_train entries where their 'best' attribute is v
         newY = []
         for i in range(len(X_train)):
             if X_train[i][best] == v:#hence, the amount of examples will shrink as the tree is made
                 #if v == 1:
                     #print(str(best) + " " + str(v) + " " + str(i) + " " + str(X_train[i]))
                     #print(str(Y_train[i]))
                     #print(str(best))
                 newX.append(X_train[i])
                 newY.append(Y_train[i])
         subtree = self.decisionTreeLearning(attributes, newX, newY, tree)
         tree.children.append(subtree)
      #print(set(self.usedBests))
      return tree

   def train(self, X_train, Y_train): 
      # receives a list of objects of type Example
      # TODO: implement decision tree training
      #https://gyazo.com/f58981e356ba1ce222d74a04829576d5

      attributes = []
      for i in range(len(X_train[0])):
         att = []
         for j in range(len(X_train)):
            att.append(X_train[j][i])
         attributes.append(att)

      return self.decisionTreeLearning(attributes, X_train, Y_train, None)

    

class Tree:
    def __init__(self, value, parent):
        self.value = value #root will be an attribute's index from the main data set
        self.children = [] #each will have 2 children, one for 0 and one for 1
        self.parent = parent
        self.depth = 1
        if parent is not None:
            self.depth = parent.depth+1

    def __str__(self):
        curr = self
        
        return("Curr: " + str(self.value)) + "\n" + "Kids: " + str(self.children[0]) + " " + str(self.children[1]) + "\n Depth: " + str(self.depth)

def testTree(X_test, Y_test, tree):
    print("# tests: " + str(len(Y_test)))
    succ = 0
    fail = 0
    for i in range(len(X_test)):
        result = testTree_case(X_test[i], Y_test[i], tree)
        if result:
            succ += 1
        elif not result:
            fail += 1
        else:
            break
    print("Succ: %i, fail: %i" %(succ, fail))
    return (succ/(succ+fail)) #% success rate

def testTree_case(data, label, tree):
    curr = tree
    while isinstance(curr, Tree):
        curr = curr.children[data[curr.value]]
    return (curr == label)

#===	   
def compute_accuracy(dt_classifier, X_test, Y_test):
   numRight = 0
   for i in range(len(Y_test)):
      x = X_test[i]
      y = Y_test[i]
      if y == dt_classifier.predict(x) :
         numRight += 1
   return (numRight*1.0)/len(Y_test)

def outputResults(fName, results):
    f = open(fName, "w")
    f.write(results)
    f.close()
    print("Finished writing.")

def predict(X, tree):
    Y = []
    for i in X:
        curr = tree
        while isinstance(curr, Tree):
            curr = curr.children[i[curr.value]]
        Y.append(curr)
    return Y

#==============================================
#==============================================
#X_train, Y_train = read_datafile(sys.argv[1])
#X_test, Y_test = read_datafile(sys.argv[4])

# TODO: write your code
trainFile = sys.argv[1]
split_random = (str(sys.argv[2]).lower() == "r")
depth_limit = int(sys.argv[3])
testFile = sys.argv[4]
outputFname = sys.argv[5]

assert os.path.isfile(trainFile), "Train data, " + trainFile + ", does not exist" #1
assert os.path.isfile(testFile), "Test data, " + testFile + ", does not exist" #4
assert str(sys.argv[2]).lower() == "r" or str(sys.argv[2]).lower() == "i", "'" +  str(sys.argv[2]) + "' is not a valid choice of splitting rule. Please enter R or I." #2
assert depth_limit >= 0, "Depth limit cannot be negative"

X_train, Y_train = read_datafile(trainFile)
X_test, Y_test = read_datafile(testFile)

main = DecisionTree(split_random, depth_limit)
decisionTree = main.train(X_train, Y_train)
succRate = testTree(X_test, Y_test, decisionTree)
outputResults(outputFname, str(succRate*100))
