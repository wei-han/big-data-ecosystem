from __future__ import print_function
import shiqi
import threading
import time
import random
import numpy as np
import tensorflow as tf
import weihan
from collections import deque

actions = shiqi.actions

gui_display = True

if(not gui_display):
	shiqi.gui_off()


GAMMA = 0.8
BATCH = 49

state_input_1 = tf.placeholder(
	tf.float32,
	[None,13,13,1])

action_input = tf.placeholder(
	tf.bool,
	shape=(BATCH,4))

reward_input = tf.placeholder(
	tf.float32,
	shape=(BATCH))

max_val_input = tf.placeholder(
	tf.float32,
	shape=(BATCH))

terminal_input = tf.placeholder(
	tf.float32,
	shape=(BATCH))

#def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    #return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


conv1_weights = tf.Variable(
  tf.truncated_normal([5, 5, 1, 16],  # 5x5 filter, depth 16.
                      stddev=0.1))
conv1_biases = tf.Variable(tf.zeros([16]))

conv2_weights = tf.Variable(
  tf.truncated_normal([4, 4, 16, 32], # 3x3 filter, depth 32
                      stddev=0.1))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[32]))

conv3_weights = tf.Variable(
  tf.truncated_normal([3, 3, 32, 64], # 3x3 filter, depth 64
                      stddev=0.1))
conv3_biases = tf.Variable(tf.constant(0.1, shape=[64]))


fc1_weights = tf.Variable(  # fully connected, depth 128.
  tf.truncated_normal([1024, 512],
                      stddev=0.1))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
fc2_weights = tf.Variable(
  tf.truncated_normal([512, 4],
                      stddev=0.1))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[4]))

def network(data):
	conv = tf.nn.conv2d(data,
                      conv1_weights,
                      strides=[1, 1, 1, 1],
                      padding='VALID')

	relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # h_pool1 = max_pool_2x2(conv)

	conv = tf.nn.conv2d(relu,
                      conv2_weights,
                      strides=[1, 1, 1, 1],
                      padding='VALID')
  	relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))

  	conv = tf.nn.conv2d(relu,
                      conv3_weights,
                      strides=[1, 1, 1, 1],
                      padding='VALID')
  	relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))

  	relu_flat = tf.reshape(relu, [-1,1024])

  	hidden = tf.nn.relu(tf.matmul(relu_flat, fc1_weights) + fc1_biases)

  	return tf.matmul(hidden, fc2_weights) + fc2_biases

sess = tf.InteractiveSession()
sess.as_default()



action_array_1 = network(state_input_1)
tt = reward_input + terminal_input * (GAMMA * max_val_input)
tt = tf.reshape(tt,(BATCH,1))
target_prep = tf.tile(tt,[1,4])
target = tf.select(action_input, target_prep, action_array_1)

Qerror = tf.sub(target, action_array_1)
loss = .5*tf.reduce_sum(tf.mul(Qerror, Qerror))

optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

saver = tf.train.Saver()
tf.initialize_all_variables().run()

checkpoint = tf.train.get_checkpoint_state("saved_networks")
if checkpoint and checkpoint.model_checkpoint_path:
	saver.restore(sess, checkpoint.model_checkpoint_path)
	print("Successfully loaded:", checkpoint.model_checkpoint_path)

def see_action(action,i,j):

	if action == actions[0]:
		reward, s2, t = shiqi.see_move(0, -1,i,j)
	elif action == actions[1]:
		reward, s2, t= shiqi.see_move(1, 0,i,j)
	elif action == actions[2]:
		reward, s2, t = shiqi.see_move(0, 1,i,j)
	elif action == actions[3]:
		reward, s2, t = shiqi.see_move(-1, 0,i,j)
	else:
		return

	return reward, s2, t

def do_action(action):

	if action == actions[0]:
		shiqi.do_move(0, -1)
	elif action == actions[1]:
		shiqi.do_move(1, 0)
	elif action == actions[2]:
		shiqi.do_move(0, 1)
	elif action == actions[3]:
		shiqi.do_move(-1, 0)
	else:
		return

def network_triangles():
	D = deque()
	for i in range(shiqi.x):
		for j in range(shiqi.y):
			state_peek_1 = shiqi.get_state((i,j))
			state_peek_1 = np.reshape(state_peek_1,(1, 13, 13, 1)).astype(np.float32)
			feed_dict = {state_input_1: state_peek_1}
			values_1 = sess.run(action_array_1, feed_dict=feed_dict)
			state_peek_1 = np.reshape(state_peek_1,(13, 13, 1)).astype(np.float32)

			random_index = np.random.choice(4,1)
			try_index = random_index[0]
			try_act = actions[try_index]

			try_act_prep = np.reshape([False, False, False, False],(4)).astype(np.bool)
			try_act_prep[try_index] = True

			reward, s2, terminal = see_action(try_act,i,j)

			state_peek_2 = np.reshape(s2,(1, 13, 13, 1)).astype(np.float32)
			feed_dict = {state_input_1: state_peek_2}
			values_2 = sess.run(action_array_1, feed_dict=feed_dict)

			max_val_data = np.amax(values_2)

			D.append((state_peek_1, try_act_prep, reward, max_val_data, terminal))

			if(gui_display):
				for action in actions:
					shiqi.set_cell_score(i,j,action,values_1)

	return D

def run():
    trials = 1
    steps = 1
    t = 0
    hit_one = True

    weihan.FloodFillValues()


    opt_steps = weihan.get_value(0,4)

    sub_trials = 1

    train = True
    limit_of_mazes = 100
    save_point = 20
    number_trial = -1
    max_steps = -1


    shiqi.set_maze_size(limit_of_mazes)

    while trials < number_trial or (number_trial == -1):

    	state_1 = shiqi.get_state(shiqi.player)



    	state_peek = np.reshape(state_1,(1, 13, 13, 1)).astype(np.float32)
    	feed_dict = {state_input_1: state_peek}

    	net_out_1 = sess.run(action_array_1, feed_dict=feed_dict)



    	max_index = np.argmax(net_out_1[0])

    	max_act = actions[max_index]

    	do_action(max_act)

    	if shiqi.has_restarted() or (steps > max_steps and max_steps > 0):

    		if(steps==opt_steps or (trials < limit_of_mazes or limit_of_mazes < 0)):
    			trials+=1
    			hit_one = True

    		if(steps < max_steps or max_steps == -1):
    			sub_trials+=1
			print(steps)
    		steps = 0

    		print('at trial {}'.format(trials))

    		shiqi.restart_game(trials)

    		weihan.FloodFillValues()

    		opt_steps = weihan.get_value(0,4)

        	if save_point > 0 and trials % save_point == 0 and hit_one:
        		saver.save(sess, 'saved_networks/' + 'async_maze' + '-dqn', global_step = t)

        		print('completed trial {}'.format(trials))

        		hit_one = False

        		sub_trials = 1

    	if(train):
    		D = network_triangles()

    		minibatch = random.sample(D, BATCH)

    		s1_update = [d[0] for d in minibatch]
    		a_update  = [d[1] for d in minibatch]
    		r_update  = [d[2] for d in minibatch]
    		mv_update = [d[3] for d in minibatch]
    		term      = [d[4] for d in minibatch]

    		feed_dict = {state_input_1: s1_update, action_input: a_update, reward_input: r_update, max_val_input: mv_update, terminal_input: term}

    		_, my_loss, start, _end_, my_tt = sess.run([optimizer, loss, action_array_1, target, tt], feed_dict=feed_dict)


    	steps += 1
    	t += 1

    if(max_steps > 0):
    	print('completed trial {}'.format(trials))
    	print('took {} subtrials'.format(sub_trials))

t = threading.Thread(target=run)
t.daemon = True
t.start()
shiqi.start_game()
