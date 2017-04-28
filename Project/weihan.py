import shiqi
import numpy as np

Values = []
walls = 0

class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)

cell_stack = Stack()

class MazeCell:
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value

def FloodFillValues():
	global walls, cell_stack, Values

	walls = shiqi.x

	Values = [[-1 for i in range(walls)] for j in range(walls)]
    #decides where the floodfill starts from
	if walls%2 == 0:
		base_1 = walls/2-1
		base_2 = walls/2

		cell_stack.push(MazeCell(base_1,base_1,0))
		cell_stack.push(MazeCell(base_2,base_1,0))
		cell_stack.push(MazeCell(base_1,base_2,0))
		cell_stack.push(MazeCell(base_2,base_2,0))

	else:
		base = (walls-1)/2
		cell_stack.push(MazeCell(base,base,0))

	while(cell_stack.size() > 0):
		current_cell = cell_stack.pop()
		SetCell(current_cell.x,current_cell.y,current_cell.value)

#set value in maze cell by x and y location
def SetCell( x, y, value):
	global Values, cell_stack
    # -1 + 1 = 0
	current_value = Values[x][y]
	if current_value == -1 or value < current_value:
		Values[x][y] = value

		if shiqi.wall_check(x,y,0,-1) and y > 0:
			cell_stack.push(MazeCell(x,y-1,value+1))

		if shiqi.wall_check(x,y,1,0) and x < walls-1:
			cell_stack.push(MazeCell(x+1,y,value+1))

		if shiqi.wall_check(x,y,0,1) and y < walls-1:
			cell_stack.push(MazeCell(x,y+1,value+1))

		if shiqi.wall_check(x,y,-1,0) and x > 0:
			cell_stack.push(MazeCell(x-1,y,value+1))

def get_value(x,y):
	return Values[x][y]
