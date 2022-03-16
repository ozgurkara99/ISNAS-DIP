from typing import Iterator, List, Iterable, Any, Optional
from collections import defaultdict


class Tree():
    '''Implements a tree structure.
    
    Attributes:
    -----------
    name: str
        Name of the tree. name attribute should be unique among childrens of a\
        tree otherwise undefined behaviour occurs.
    data: Any
        Data stored in the tree.
        
    Methods:
    --------
    add_child(self, child: Tree) -> None:
        Adds the given Tree object as a child.
    children(self) -> List[Tree]:
        Returns the children of the tree as a list.
    __getitem__(self, name: str) -> Tree:
        Access children by their names.
    __contains__(self, name: str) -> bool:
        Check whether the tree has a children with the given name.
    '''
    
    def __init__(self, name: str, data: Any, children: Iterable['Tree'] = None) -> None:
        '''Initializes a tree object.
        
        Args:
            name: Name of the tree. name attribute should be unique among \
                childrens of a tree otherwise undefined behaviour occurs.
            data: Data to be stored in the tree.
            children: Children of the tree.'''
        
        if children is None:
            children = []
        
        self.name = name
        self.data = data
        
        # we store the children as a dictionary instead of list for fast access
        # to children by their names
        self.__children = {child.name: child for child in children}
    
    def __getitem__(self, name: str) -> 'Tree':
        '''Access the children of the tree by their name.'''
        
        return self.__children[name]
    
    def __contains__(self, name: str) -> bool:
        '''Check whether the tree has a children with the given name.'''
        
        return name in self.__children
    
    def add_child(self, child: 'Tree') -> None:
        '''Add a tree object as a child.'''
        
        self.__children[child.name] = child
    
    def children(self) -> List['Tree']:
        '''Get the children of the tree as a list.'''
        
        return list(self.__children.values())

class ModDefaultDict(defaultdict):
    '''A modified defaultdict. The only difference between the usual \
        defaultdict is that the default constructor takes the missing as an \
        input. So the default constructor should be a callable that takes one \
        argument.'''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __missing__(self, key: str):
        result = self.default_factory(key)
        self.__setitem__(key, result)
        return result
    
class Queue():
    '''
    A simple fixed sized queue implementation. The elements are pushed and \
    popped like a usual queue but if the queue is full, an element is \
    popped before the new element is pushed into the queue.
        
    Attributes:
    -----------
        size: int
            Size of the queue. If it is None, size of the queue is infinite.
        data: List
            Data stored in the queue.
    
    Methods:
    --------
        push(self, item: Any) -> None
            Add an item to the queue. If the queue is full, then an element is \
            popped from the queue.
            
        pop(self, item: Any) -> Any
            Pop an element from the queue.
        
        top(self) -> Any
            Get the oldest element in the queue.
            
        bottom(self) -> Any
            Get the newest element in the queue.
            
        empty(self) -> bool
            See whether the queue is empty or not.
        
        __len__(self) -> int
            Get the number of items in the queue.
        
        __getitem__(self, index: int) -> Any
            Get the item of the given index from the queue.
            
        __iter__(self) -> Iterator:
            Get an iterator for the queue.'''
    
    def __init__(self, size: Optional[int] = None) -> None:
        
        self.size = size
        self.data = []
        
    def push(self, item: Any) -> None:
        
        self.data = [item] + self.data
        
        if self.size is not None and len(self.data) > self.size:
            self.data.pop()
    
    def pop(self) -> Any:
        return self.data.pop()
    
    def top(self) -> Any:
        return self.data[0]
    
    def bottom(self) -> Any:
        return self.data[-1]
    
    def empty(self) -> bool:
        return len(self.data) == 0
    
    def __len__(self) -> Optional[int]:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Any:
        return self.data[index]
    
    def __iter__(self) -> Iterator:
        return iter(self.data)