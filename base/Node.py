class Node:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def isLeaf(self):
        return self.left is None and self.right is None
