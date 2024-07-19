# username - mayamarom
# id1      - 209162544
# name1    - Maya Marom
# id2      - 207004078
# name2    - Keshet Brandt

import random

"""A class represnting a node in an AVL tree"""


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

        @type value: str or None
        @param value: data of your node
        """

    def __init__(self, value, real=True, parent=None, height=0, size=1, BF=0):
        self.value = value
        self.parent = parent
        self.height = height
        self.size = size
        self.BF = BF
        self.real = real #False if virtual
        if real:
            self.left = AVLNode(None, False, self, -1, 0, -1)  # Virtual Node
            self.right = AVLNode(None, False, self, -1, 0, -1)  # Virtual Node
        else:
            self.left = None  # Virtual Node doesn't have left
            self.right = None  # Virtual Node doesn't have right

    """returns the left child
        @rtype: AVLNode
        @returns: the left child of self, AVLNode with value None (virtual) if there is no left child
        """

    def getLeft(self):  # O(1)
        if not self.isRealNode():
            return None
        return self.left

    """returns the right child

        @rtype: AVLNode
        @returns: the right child of self, AVLNode with value None (virtual) if there is no right child
        """

    def getRight(self):  # O(1)
        if not self.isRealNode():
            return None
        return self.right

    """returns the parent 

        @rtype: AVLNode
        @returns: the parent of self, None if there is no parent
        """

    def getParent(self):  # O(1)
        return self.parent

    """return the value

        @rtype: str
        @returns: the value of self, None if the node is virtual
        """

    def getValue(self):  # O(1)
        return self.value

    """returns the height

        @rtype: int
        @returns: the height of self, -1 if the node is virtual
        """

    def getHeight(self):  # O(1)
        return self.height

    """sets left child

        @type node: AVLNode
        @param node: a node
        """

    def setLeft(self, node):  # O(1)
        self.left = node

    """sets right child

        @type node: AVLNode
        @param node: a node
        """

    def setRight(self, node):  # O(1)
        self.right = node

    """sets parent

        @type node: AVLNode or None 
        @param node: a node or None 
        """

    def setParent(self, node):  # O(1)
        self.parent = node

    """sets value

        @type value: str
        @param value: data
        """

    def setValue(self, value):  # O(1)
        self.value = value

    """sets the balance factor of the node

        @type h: int
        @param h: the height
        """

    def setHeight(self, h):  # O(1)
        self.height = h

    """returns whether self is not a virtual node 

        @rtype: bool
        @returns: False if self is a virtual node, True otherwise.
        """

    def isRealNode(self):  # O(1)
        return self.real

    """Functions we created:
        Sets the size of the node
        """

    def setSize(self, newsize):  # O(1)
        self.size = newsize

    """returns the size
        @rtype: int
        @returns: the size of self, 0 if the node is virtual
        """

    def getSize(self):  # O(1)
        return self.size

    """returns the Balance Factor
        @rtype: int
        @returns: the balance factor of self, -1 if the node is virtual
        """

    def getBF(self):  # O(1)
        return self.BF

    """sets the Balance Factor according to the children's height's
        """

    def setBF(self):  # O(1)
        if not self.real:
            self.BF = -1
        self.BF = self.getLeft().getHeight() - self.getRight().getHeight()


"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
    """
        Constructor, you are allowed to add more fields.  

        """

    def __init__(self, node=None):
        self.root = node
        self.Last = node #First index of list 
        self.First = node #last index of list 
        if self.root is None:
            self.size = 0
        else:
            self.size = self.getRoot().getSize()

    """returns whether the list is empty

        @rtype: bool
        @returns: True if the list is empty, False otherwise
        """

    def empty(self):  # O(1)
        if self.root is None or self is None:
            return True
        return False

    """retrieves the value of the i'th item in the list

        @type i: int
        @pre: 0 <= i < self.length()
        @param i: index in the list
        @rtype: str
        @returns: the the value of the i'th item in the list
        """

    def retrieve(self, i):  # O(logn)
        if i > self.length() or i < 0:
            return None
        if self.empty():
            return None
        return self.Select(i + 1).getValue()

    """inserts val at position i in the list

        @type i: int
        @pre: 0 <= i <= self.length()
        @param i: The intended index in the list to which we insert val
        @type val: str
        @param val: the value we inserts
        @rtype: list
        @returns: the number of rebalancing operation due to AVL rebalancing
        """

    def insert(self, i, val):  # O(logn)
        if self.empty():
            newRoot = AVLNode(val, True, None)
            self.root = newRoot
            self.First = newRoot
            self.Last = newRoot
            self.size = 1
            return 0
        n = self.length()
        if i == n:
            maxi = self.Last
            temp = AVLNode(val, True, maxi)
            maxi.setRight(temp)
            temp.setParent(maxi)
            self.UpdateSizes(temp)
            self.UpdateHeights(temp)
        elif i < n:
            if i == 0:
                place = AVLNode(val, True, self.First)
                self.First.setLeft(place)
                place.setParent(self.First)
            else:
                place = self.Select(i + 1)
                if not place.getLeft().isRealNode():
                    place.setLeft(AVLNode(val, True, place))
                    place.getLeft().setParent(place)
                    i = self.Rank(place)
                else:
                    pre = self.predecessor(place)
                    pre.setRight(AVLNode(val, True, pre))
                    pre.getRight().setParent(pre)
                    place = pre
            self.UpdateSizes(place)
            self.UpdateHeights(place)
        self.Last = self.Select(self.size)
        self.First = self.Select(1)
        return self.Balance(i)

    """returns the successor of node
        @rtype: node
        @returns: the successor of node, None if node is last
        """

    def successor(self, node):  # O(logn)
        if self.Last == node:
            return None
        if node.getRight().isRealNode():
            return self.minNode(node.getRight())
        parent = node.getParent()
        while parent is not None and node == parent.getRight():
            node = parent
            parent = node.getParent()
        return parent

    """returns the predecessor of node
        @rtype: node
        @returns: the predecessor of node, None if node is first 
        """

    def predecessor(self, node):  # O(logn)
        if self.First == node:
            return None
        if node.getLeft().isRealNode():
            return self.maxNode(node.getLeft())
        parent = node.getParent()
        while parent is not None and node == parent.getLeft():
            node = parent
            parent = node.getParent()
        return parent

    """returns the node in index i in the list 
        @rtype: node, None if the list is empty 
        """

    def Select(self, i): #O(logn)
        if self.empty():
            return None
        return self.TreeSelect(i, self.root)

    """recursive call for select 
        """

    def TreeSelect(self, i, node):
        if not node.getLeft().isRealNode():
            temp = 1
        else:
            temp = node.getLeft().getSize() + 1
        if i == temp:
            return node
        elif i < temp:
            return self.TreeSelect(i, node.getLeft())
        else:
            return self.TreeSelect(i - temp, node.getRight())

    """returns the index of the node in the list, -1 if the list is empty 
        @rtype: int
        """

    def Rank(self, node): #O(logn)
        if self.empty():
            return -1
        if self.getRoot().getSize() == 1:
            return 1
        temp = node.getLeft().getSize() + 1
        son = node
        while son != self.getRoot():
            if son == son.getParent().getRight():
                temp = temp + son.getParent().getLeft().getSize() + 1
            son = son.getParent()
        return temp

    """Balances the tree
        @rtype: int
        @returns: number of rotations done 
        """

    def Balance(self, i):  #O(logn)
        if self.empty():
            return 0
        cnt = 0
        if i == 0:
            cur = self.First
        else:
            cur = self.Select(i)
        cur.setBF()
        while cur != self.root:
            if cur.getBF() == 2 or cur.getBF() == -2:
                cnt += self.Rotate(cur)
                cur.setBF()
            cur = cur.getParent()
            cur.setBF()
        if cur.getBF() == 2 or cur.getBF() == -2:
            cnt += self.Rotate(cur)
            cur.setBF()
        self.root.setBF()
        return cnt

    """Updates the size field from the node to the root 
        """

    def UpdateSizes(self, node): #O(logn)
        while node != self.root.getParent():
            node.setSize(node.getLeft().getSize() + node.getRight().getSize() + 1)
            node = node.getParent()
        self.size = self.getRoot().getSize()

    """Updates the height field from the node to the root
        """

    def UpdateHeights(self, node): #O(logn)
        while self.root.getParent() != node:
            node.setHeight(max(node.getLeft().getHeight(), node.getRight().getHeight()) + 1)
            node = node.getParent()

    """Rotates the tree in order to balance  
        @rtype: number of rotations done 
        """

    def Rotate(self, node):  #O(logn)
        if node.getBF() == 2:
            node.getLeft().setBF()
            if node.getLeft().getBF() == 1 or node.getLeft().getBF() == 0:
                self.RotateRight(node.getLeft())
                if node == self.root:
                    self.root = node.getParent()
                    self.root.setHeight(max(self.root.getLeft().getHeight(), self.root.getRight().getHeight()) + 1)
                return 1
            else:
                self.RotateLeft(node.getLeft())
                self.RotateRight(node.getLeft())
                if node == self.root:
                    self.root = node.getParent()
                    self.root.setHeight(max(self.root.getLeft().getHeight(), self.root.getRight().getHeight()) + 1)
                return 2
        else:
            node.getRight().setBF()
            if node.getRight().getBF() == -1 or node.getRight().getBF() == 0:
                self.RotateLeft(node.getRight())
                if node == self.root:
                    self.root = node.getParent()
                    self.root.setHeight(max(self.root.getLeft().getHeight(), self.root.getRight().getHeight()) + 1)
                return 1
            else:
                self.RotateRight(node.getRight())
                self.RotateLeft(node.getRight())
                if node == self.root:
                    self.root = node.getParent()
                    self.root.setHeight(max(self.root.getLeft().getHeight(), self.root.getRight().getHeight()) + 1)
                return 2

    """Right rotation at node  
        """

    def RotateRight(self, node):  #O(logn)
        middle = node
        First = node.getParent()
        Last = node.getLeft()
        if First.getLeft() == middle:
            parent = First.getParent()
            First.setLeft(middle.getRight())
            middle.getRight().setParent(First)
            middle.setRight(First)
            First.setParent(middle)
            if parent is not None:
                if parent.getLeft() == First:
                    parent.setLeft(middle)
                else:
                    parent.setRight(middle)
            middle.setParent(parent)
            First.setSize(First.getLeft().getSize() + First.getRight().getSize() + 1)
            middle.setSize(middle.getLeft().getSize() + middle.getRight().getSize() + 1)
            self.UpdateHeights(First)
        else:
            middle.setLeft(Last.getRight())
            Last.getRight().setParent(middle)
            Last.setRight(middle)
            middle.setParent(Last)
            Last.setParent(First)
            First.setRight(Last)
            middle.setSize(middle.getLeft().getSize() + middle.getRight().getSize() + 1)
            Last.setSize(Last.getLeft().getSize() + Last.getRight().getSize() + 1)
            self.UpdateHeights(middle)

    """Left rotation at node  
        """

    def RotateLeft(self, node): #O(logn)
        middle = node
        First = node.getParent()
        Last = node.getRight()
        if First.getLeft() == middle:
            First.setLeft(Last)
            Last.setParent(First)
            middle.setRight(Last.getLeft())
            Last.getLeft().setParent(middle)
            Last.setLeft(middle)
            middle.setParent(Last)
            Last.setSize(Last.getLeft().getSize() + Last.getRight().getSize() + 1)
            middle.setSize(middle.getLeft().getSize() + middle.getRight().getSize() + 1)
            self.UpdateHeights(middle)
        else:
            parent = First.getParent()
            First.setRight(middle.getLeft())
            middle.getLeft().setParent(First)
            middle.setLeft(First)
            First.setParent(middle)
            middle.setParent(parent)
            if parent is not None:
                if parent.getLeft() == First:
                    parent.setLeft(middle)
                else:
                    parent.setRight(middle)
            First.setSize(First.getLeft().getSize() + First.getRight().getSize() + 1)
            middle.setSize(middle.getLeft().getSize() + middle.getRight().getSize() + 1)
            self.UpdateHeights(First)

    """deletes the i'th item in the list

        @type i: int
        @pre: 0 <= i < self.length()
        @param i: The intended index in the list to be deleted
        @rtype: int
        @returns: the number of rebalancing operation due to AVL rebalancing
        """

    def delete(self, i): #O(logn)
        if self.empty():
            return -1
        if i > self.getRoot().getSize() or i < 0:
            return -1
        node = self.Select(i + 1)
        if node == self.root:
            if node.getSize() == 1:
                self.root = None
                self.First = None
                self.Last = None
                self.size = 0
                return 0
            else:
                if not node.getRight().isRealNode():
                    self.root = node.getLeft()
                    self.UpdateSizes(self.root)
                    self.UpdateHeights(self.root)
                    self.Last = self.Select(self.size)
                    self.First = self.Select(1)
                    return 0
                else:
                    suc = self.successor(node)
                    sucParent = suc.getParent()
                    fixes = suc
                    if sucParent != node:
                        sucParent.setLeft(suc.getRight())
                        sucParent.getLeft().setParent(sucParent)
                        suc.setRight(node.getRight())
                        suc.getRight().setParent(suc)
                        fixes = sucParent
                    else:
                        node.setRight(AVLNode(None, False, node, -1, 0, -1))
                    suc.setLeft(node.getLeft())
                    node.getLeft().setParent(suc)
                    suc.setParent(None)
                    self.root = suc
                    self.UpdateSizes(fixes)
                    self.UpdateHeights(fixes)
                    self.Last = self.Select(self.size)
                    self.First = self.Select(1)
                    return self.Balance(self.Rank(fixes))
        if not node.getLeft().isRealNode() and not node.getRight().isRealNode():
            if node.getParent().getLeft() == node:
                node.getParent().setLeft(AVLNode(None, False, node.getParent(), -1, 0, -1))
            else:
                node.getParent().setRight(AVLNode(None, False, node.getParent(), -1, 0, -1))
            fixes = node.getParent()
            self.UpdateSizes(fixes)
            self.UpdateHeights(fixes)
        elif node == self.Last:
            fixes = node.getLeft()
            node.getParent().setRight(fixes)
            fixes.setParent(node.getParent())
            self.UpdateSizes(fixes)
            self.UpdateHeights(fixes)
            fixes = fixes.getParent()
            self.Last = self.Select(self.size)
            self.First = self.Select(1)
            return self.Balance(self.Rank(fixes))
        elif node.getLeft().isRealNode() and not node.getRight().isRealNode():
            parent = node.getParent()
            if parent.getRight() == node:
                parent.setRight(node.getLeft())
                node.getLeft().setParent(parent)
            else:
                parent.setLeft(node.getLeft())
                node.getLeft().setParent(parent)
            fixes = parent
            self.UpdateSizes(fixes)
            self.UpdateHeights(fixes)
        else:
            suc = self.successor(node)
            fixes = suc
            parent = node.getParent()
            if parent == suc:
                parent.setLeft(node.getLeft())
                node.getLeft().setParent(parent)
            else:
                if suc.getLeft().isRealNode():
                    sucParent = suc.getParent()
                    if sucParent is not None:
                        sucParent.setLeft(suc.getLeft())
                    else:
                        self.root = suc.getLeft()
                    suc.getLeft().setParent(sucParent)
                elif suc.getRight().isRealNode() and suc.getParent() != node:
                    sucParent = suc.getParent()
                    if sucParent is not None:
                        sucParent.setLeft(suc.getRight())
                        fixes = sucParent
                    suc.getRight().setParent(sucParent)
                elif suc.getParent() != node:
                    suc.getParent().setLeft(AVLNode(None, False, suc.getParent(), -1, 0, -1))
                    fixes = suc.getParent()
                if parent.getRight() == node:
                    parent.setRight(suc)
                    if parent.getParent() == suc:
                        grandparent = suc.getParent()
                        parent.setParent(grandparent)
                        if grandparent is not None:
                            if grandparent.getLeft() == suc:
                                grandparent.setLeft(parent)
                            else:
                                grandparent.setRight(parent)
                else:
                    parent.setLeft(suc)
                if node.getRight() != suc:
                    suc.setRight(node.getRight())
                    node.getRight().setParent(suc)
                suc.setParent(parent)
                suc.setLeft(node.getLeft())
                suc.getLeft().setParent(suc)
            self.UpdateSizes(fixes)
            self.UpdateHeights(fixes)
        self.Last = self.Select(self.size)
        self.First = self.Select(1)
        return self.Balance(self.Rank(fixes))

    """returns the value of the first item in the list

        @rtype: str
        @returns: the value of the first item, None if the list is empty
        """

    def first(self): #O(1)
        if self.First is None:
            return None
        return self.First.getValue()

    """returns the value of the last item in the list

        @rtype: str
        @returns: the value of the last item, None if the list is empty
        """

    def last(self): #O(1)
        if self.Last is None:
            return None
        return self.Last.getValue()

    """returns an array representing list 

        @rtype: list
        @returns: a list of strings representing the data structure
        """

    def listToArray(self):  #O(n)
        lst = []
        if self.empty():
            return lst
        self.listToArrayMem(self.getRoot(), lst)
        return lst

    """ListToArray recursive call
        """

    def listToArrayMem(self, node, temp):
        if not node.isRealNode():
            return
        self.listToArrayMem(node.getLeft(), temp)
        temp.append(node.getValue())
        self.listToArrayMem(node.getRight(), temp)

    """returns the size of the list 

        @rtype: int
        @returns: the size of the list
        """

    def length(self): #O(1)
        if self.empty():
            return 0
        return self.getRoot().getSize()

    """sort the info values of the list
        @rtype: list
        @returns: an AVLTreeList where the values are sorted by the info of the original list.
        """

    def mergesort(self, lst): #O(nlogn)
        n = len(lst)
        if n <= 1:
            return lst
        else:
            return self.merge(self.mergesort(lst[:n // 2]), self.mergesort(lst[n // 2::]))

    def merge(self, lst1, lst2): #O(n)
        temp = [None for i in range(len(lst1) + len(lst2))]
        index1 = 0
        index2 = 0
        count = 0
        nones = 1
        while index1 < len(lst1) and index2 < len(lst2):
            if lst1[index1] is None:
                temp[len(lst1) + len(lst2) - nones] = None
                index1 += 1
                nones += 1
            elif lst2[index2] is None:
                temp[len(lst1) + len(lst2) - nones] = None
                index2 += 1
                nones += 1
            elif lst1[index1] < lst2[index2]:
                temp[count] = lst1[index1]
                index1 += 1
                count += 1
            else:
                temp[count] = lst2[index2]
                index2 += 1
                count += 1
        if len(lst1) == index1:
            while index2 < len(lst2):
                temp[count] = lst2[index2]
                index2 += 1
                count += 1
        else:
            while index1 < len(lst1):
                temp[count] = lst1[index1]
                index1 += 1
                count += 1
        return temp

    def sort(self): #O(nlogn)
        if self.empty():
            return self
        temp = self.listToArray()
        lexi = self.mergesort(temp)
        return self.buildTree(lexi)

    """Builds a tree from python list 
        @rtype: list
        @returns: AVLTreeList
        """

    def buildTree(self, lst): #O(nlogn)
        root = self.buildTreeMem(lst)
        tree = AVLTreeList()
        tree.root = root
        tree.First = self.Select(1)
        tree.Last = self.Select(len(lst))
        return tree

    def buildTreeMem(self, lst):
        if len(lst) == 0:
            return AVLNode(None, False, None, -1, 0, -1)
        if len(lst) == 1:
            return AVLNode(lst[0], True, None)
        mid = len(lst) // 2
        parent = AVLNode(lst[mid], True, None)
        lst1 = lst[0:mid]
        lst2 = lst[mid + 1:]
        small = self.buildTreeMem(lst1)
        big = self.buildTreeMem(lst2)
        small.setParent(parent)
        parent.setLeft(small)
        big.setParent(parent)
        parent.setRight(big)
        self.UpdateSizes(parent)
        self.UpdateHeights(parent)
        parent.setHeight(max(small.getHeight(), big.getHeight()) + 1)
        return parent

    """permute the info values of the list 

        @rtype: list
        @returns: an AVLTreeList where the values are permuted randomly by the info of the original list.
         ##Use Randomness
        """

    def permutation(self): #O(n)
        if self.empty():
            return self
        structure = self.copyTree()
        lst = self.listToArray()
        for i in range(len(lst)- 1, 1, -1):
            x = random.randrange(0, i)
            lst[x] , lst[i] = lst[i] , lst[x]
        structure.shuffleTree(structure.getRoot(), lst)
        structure.First = structure.Select(1)
        structure.Last = structure.Select(structure.size)
        return structure

    """Makes a copy of the list 
        @returns: an AVLTreeList 
        """

    def copyTree(self): #O(n)
        root = self.copyTreeMem(self.getRoot())
        copy = AVLTreeList()
        copy.root = root
        copy.size = self.root.size
        return copy

    def copyTreeMem(self, node):
        if not node.isRealNode():
            return node
        copyNode = AVLNode(node.getValue(), node.isRealNode(), None, node.getHeight(), node.getSize(), node.getBF())
        copyNode.setLeft(self.copyTreeMem(node.getLeft()))
        copyNode.getLeft().setParent(copyNode)
        copyNode.setRight(self.copyTreeMem(node.getRight()))
        copyNode.getRight().setParent(copyNode)
        return copyNode

    """Shuffles the list  
        """

    def shuffleTree(self, node, lst): #O(n)
        if not node.isRealNode():
            return
        self.shuffleTree(node.getLeft(), lst)
        val = lst[len(lst) -1]
        node.setValue(val)
        lst.pop(len(lst) -1)
        self.shuffleTree(node.getRight(), lst)

    """concatenates lst to self

        @type lst: AVLTreeList
        @param lst: a list to be concatenated after self
        @rtype: int
        @returns: the absolute value of the difference between the height of the AVL trees joined
        """

    def concat(self, lst):  #O(logn)
        if lst.empty() and self.empty():
            lst.root = self.root
            lst.First = self.First
            lst.Last = self.Last
            lst.size = self.size
            return 0
        if lst.empty():
            lst.root = self.root
            lst.First = self.First
            lst.Last = self.Last
            lst.size = self.size
            return self.getRoot().getHeight()
        if self.empty():
            self.root = lst.root
            self.First = lst.First
            self.Last = lst.Last
            self.size = lst.size
            return lst.getRoot().getHeight()

        heightOriginal = self.getRoot().getHeight()
        heightNew = lst.getRoot().getHeight()

        if heightOriginal >= heightNew:
            nodeH = self.getRoot()
            while nodeH.getHeight() > heightNew and nodeH.getRight().isRealNode():
                nodeH = nodeH.getRight()
            nodeHMax = self.maxNode(nodeH)
            fixes = nodeHMax.getParent()
            if nodeHMax == nodeH:
                nodeH.setRight(lst.getRoot())
                lst.getRoot().setParent(nodeH)
                fixes = nodeHMax
            else:
                if nodeHMax.getParent() is not None:
                    nodeHMax.getParent().setRight(nodeHMax.getRight())
                nodeHMax.getRight().setParent(nodeHMax.getParent())
                if heightOriginal == heightNew:
                    nodeHMax.setParent(None)
                    self.root = nodeHMax
                else:
                    nodeHParent = nodeH.getParent()
                    if nodeHParent is not None:
                        nodeHParent.setRight(nodeHMax)
                    nodeHMax.setParent(nodeHParent)
                nodeHMax.setLeft(nodeH)
                nodeH.setParent(nodeHMax)
                nodeHMax.setRight(lst.getRoot())
                lst.getRoot().setParent(nodeHMax)
        else:
            nodeH = lst.getRoot()
            while nodeH.getHeight() > heightOriginal and nodeH.getLeft().isRealNode():
                nodeH = nodeH.getLeft()
            nodeHMin = lst.minNode(nodeH)
            fixes = nodeHMin.getParent()
            if nodeHMin == nodeH:
                nodeH.setLeft(self.getRoot())
                self.getRoot().setParent(nodeH)
                self.root = lst.getRoot()
                fixes = nodeHMin
            else:
                if nodeHMin.getParent() is not None:
                    nodeHMin.getParent().setLeft(nodeHMin.getLeft())
                nodeHMin.getLeft().setParent(nodeHMin.getParent())
                nodeHParent = nodeH.getParent()
                if nodeHParent is not None:
                    nodeHParent.setLeft(nodeHMin)
                nodeHMin.setParent(nodeHParent)
                nodeHMin.setRight(nodeH)
                nodeHMin.setLeft(self.root)
                nodeH.setParent(nodeHMin)
                self.getRoot().setParent(nodeHMin)
                self.root = lst.getRoot()
        self.UpdateSizes(fixes)
        self.UpdateHeights(fixes)
        self.Balance(self.Rank(fixes))
        self.First = self.Select(1)
        self.Last = self.Select(self.size)
        lst.Last = self.Last
        lst.root = self.root
        lst.First = self.First
        res = heightOriginal - heightNew
        if res < 0:
            return -1 * res
        return res

    """searches for a *value* in the list

        @type val: str
        @param val: a value to be searched
        @rtype: int
        @returns: the first index that contains val, -1 if not found.
        """

    def search(self, val): #O(n)
        if self.empty():
            return -1
        temp = self.listToArray()  # O(n)
        for i in range(len(temp)):  # O(n)
            if temp[i] == val:
                return i
        return -1

    """searches for the Max of sub-tree of specific node 
        @rtype: AVLNode
        @returns: the max node of subtree
        """

    def maxNode(self, node): #O(logn)
        while node.getRight().isRealNode():
            node = node.getRight()
        return node

    """searches for the Min of sub-tree of specific node 
        @rtype: AVLNode
        @returns: the min node of subtree
        """

    def minNode(self, node): #O(logn)
        while node.getLeft().isRealNode():
            node = node.getLeft()
        return node

    """returns the root of the tree representing the list
        @rtype: AVLNode
        @returns: the root, None if the list is empty
        """

    def getRoot(self): #O(1)
        return self.root

