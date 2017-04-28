from Tkinter import *
import chengli
import numpy as np
master = Tk()

wall_width = 60
pip_width = 4
(x, y) = (7, 7)
actions = ["up", "right", "down", "left"]
gui_display = True

board = Canvas(master, width=(x+1)*pip_width+x*wall_width, height=(y+1)*pip_width+y*wall_width)
player = (0, 0)
restart = False
g7 = 0
cell_scores = {}
triangle_size = 0

rows, columns = chengli.generate(7,0)
goal = (6,0)

def create_triangle(i, j, action):
	if action == actions[0]:
		return board.create_polygon((i+0.5-triangle_size)*wall_width+(i+1)*pip_width, (j+triangle_size)*wall_width+(j+1)*pip_width,
                                    (i+0.5+triangle_size)*wall_width+(i+1)*pip_width, (j+triangle_size)*wall_width+(j+1)*pip_width,
                                    (i+0.5)*wall_width+(i+1)*pip_width, j*wall_width+(j+1)*pip_width,
                                    fill="green", width=1)
	elif action == actions[2]:
		return board.create_polygon((i+0.5-triangle_size)*wall_width+(i+1)*pip_width, (j+1-triangle_size)*wall_width+(j+1)*pip_width,
                                    (i+0.5+triangle_size)*wall_width+(i+1)*pip_width, (j+1-triangle_size)*wall_width+(j+1)*pip_width,
                                    (i+0.5)*wall_width+(i+1)*pip_width, (j+1)*wall_width+(j+1)*pip_width,
                                    fill="green", width=1)
	elif action == actions[3]:
		return board.create_polygon((i+triangle_size)*wall_width+(i+1)*pip_width, (j+0.5-triangle_size)*wall_width+(j+1)*pip_width,
                                    (i+triangle_size)*wall_width+(i+1)*pip_width, (j+0.5+triangle_size)*wall_width+(j+1)*pip_width,
                                    i*wall_width+(i+1)*pip_width, (j+0.5)*wall_width+(j+1)*pip_width,
                                    fill="green", width=1)
	elif action == actions[1]:
		return board.create_polygon((i+1-triangle_size)*wall_width+(i+1)*pip_width, (j+0.5-triangle_size)*wall_width+(j+1)*pip_width,
                                    (i+1-triangle_size)*wall_width+(i+1)*pip_width, (j+0.5+triangle_size)*wall_width+(j+1)*pip_width,
                                    (i+1)*wall_width+(i+1)*pip_width, (j+0.5)*wall_width+(j+1)*pip_width,
                                    fill="green", width=1)

def render_grid():
	global walls, Width, x, y, player
	board.create_rectangle(0, 0, (x+1)*pip_width+x*wall_width, (y+1)*pip_width+y*wall_width, fill="#79CDEC", width=1)
	for i in range(x+1):
		for j in range(y+1):

			temp = {}
			for action in actions:
				temp[action] = create_triangle(i, j, action)
			cell_scores[(i,j)] = temp

			board.create_rectangle(i*pip_width+i*wall_width, j*pip_width+j*wall_width, (i+1)*pip_width+i*wall_width, (j+1)*pip_width+j*wall_width, fill="yellow", width=1)

	for n in range(len(rows)):
		for i in range(len(rows[n])):
			if rows[n][i] is 1:
				board.create_rectangle((i+1)*pip_width+i*wall_width, n*pip_width+n*wall_width, (i+1)*pip_width+(i+1)*wall_width, (n+1)*pip_width+n*wall_width, fill="black", width=1)

	for n in range(len(columns)):
		for i in range(len(columns[n])):
			if columns[n][i] is 1:
				board.create_rectangle(i*pip_width+i*wall_width, (n+1)*pip_width+n*wall_width, (i+1)*pip_width+i*wall_width, (n+1)*pip_width+(n+1)*wall_width, fill="black", width=1)

	board.grid(row=0, column=0)

def set_cell_score(i, j, action, vals):

	triangle = cell_scores[(i,j)][action]
	if(i==6 and j==6):
		board.itemconfigure(triangle, fill='blue')
		return
	if action == 'up':
		vact = 0
	elif action == 'right':
		vact = 1
	elif action == 'down':
		vact = 2
	elif action == 'left':
		vact = 3
	val = vals[0][vact]

	cell_score_min = np.min(vals)
	cell_score_max = np.max(vals)
	green_dec = int(min(255, max(0, (val - cell_score_min) * 255.0 / (cell_score_max - cell_score_min))))
	green = hex(green_dec)[2:]
	red = hex(255-green_dec)[2:]
	if len(red) == 1:
		red += "0"
	if len(green) == 1:
		green += "0"
	color = "#" + red + green + "00"
	board.itemconfigure(triangle, fill=color)

def render_player():
	global g7
	g7 = board.create_rectangle((player[0]+1)*pip_width+player[0]*wall_width+wall_width*1/3, (player[1]+1)*pip_width+player[1]*wall_width+wall_width*1/3,
			(player[0]+1)*pip_width+player[0]*wall_width+wall_width*2/3, (player[1]+1)*pip_width+player[1]*wall_width+wall_width*2/3, fill="grey", width=1, tag="g7")

if(gui_display):
	render_grid()
	render_player()

def do_move(dx, dy):

	global player, g7, restart
	if restart == True:
		restart_game()
	new_x = player[0] + dx
	new_y = player[1] + dy
	if (new_x >= 0) and (new_x < x) and (new_y >= 0) and (new_y < y) and wall_check( player[0], player[1], dx, dy):
		if(gui_display):
			board.coords(g7, (new_x+1)*pip_width+new_x*wall_width+wall_width*1/3, (new_y+1)*pip_width+new_y*wall_width+wall_width*1/3,
													(new_x+1)*pip_width+new_x*wall_width+wall_width*2/3, (new_y+1)*pip_width+new_y*wall_width+wall_width*2/3)
		player = (new_x, new_y)

		if new_x == goal[0] and new_y == goal[0]:
			print "Maze Clear"
			restart = True

def see_move(dx, dy, i, j):

	score = 0
	new_x = i + dx
	new_y = j + dy
	score += -0.1
	terminal = 1
	if (new_x >= 0) and (new_x < x) and (new_y >= 0) and (new_y < y) and wall_check( i, j, dx, dy):

		state = get_state((new_x, new_y))

		if new_x == goal[0] and new_y == goal[0]:
			score -= -0.1
			score += 10
			terminal = 0
	else:
		score -= -0.1
		score += -1.0
		state = get_state((i,j))

	return score, state, terminal

def get_state(position):
	global x, rows, columns
	state = []
	dim = 2*x-1
# all 0, peg=-1, position=0, columns, rows
	state = [[0.0 for i in range(dim)] for j in range(dim)]

	for j in np.arange(1,dim-1,2):
		for i in np.arange(1,dim-1,2):
			state[j][i] = -1

	state[position[1]*2][position[0]*2] = 1

	for j in np.arange(1,dim-1,2):
		for i in np.arange(0,dim,2):
			state[j][i] = -1*rows[j/2+1][i/2]

	for j in np.arange(0,dim,2):
		for i in np.arange(1,dim-1,2):
			state[j][i] = -1*columns[j/2][i/2+1]

	return state

def get_pos_from_state(state):
	state = np.reshape(state,(13,13))

	x, y = np.unravel_index(np.argmax(state), np.shape(state))

	return x/2, y/2

def wall_check(curr_x, curr_y, dx, dy):

	if(dx > 0):
		if columns[curr_y][curr_x+1] is not 1:
			return True

	elif(dx < 0):
		if columns[curr_y][curr_x] is not 1:
			return True

	elif(dy < 0):
		if rows[curr_y][curr_x] is not 1:
			return True

	else:
		if rows[curr_y+1][curr_x] is not 1:
			return True
	return False

def call_up(event):
	try_move(0, -1)

def call_right(event):
	try_move(1, 0)

def call_down(event):
	try_move(0, 1)

def call_left(event):
	try_move(-1, 0)

def set_maze_size(size):
	if(size > 0):
		chengli.set_maze_size(size)

def restart_game(trial):

	global player, g7, restart, rows, columns

	rows, columns = rows, columns = chengli.generate(7,trial)
	if(gui_display):
		render_grid()
		render_player()

	player = (0, 0)
	restart = False

def has_restarted():
	return restart

def gui_off():
	global gui_display
	gui_display = False

def start_game():
	master.mainloop()
