# Python bot for the four-in-a-row AI challenge at www.theaigames.com
# by Jeroen Delcour <jeroendecour@gmail.com>
#
# Built upon Lukas Knoepfel's <shylux@gmail.com> starter bot (version 1.0, 30 Mar 2016)
# under the MIT license (http://opensource.org/licenses/MIT)

from sys import stdin, stdout
import numpy as np
from KnuthMorrisPratt import KnuthMorrisPratt
import time
# import # logging, sys
# logging.basicConfig(filename='JDbot.log',format='%(asctime)s %(levelname)s: %(message)s',level=# logging.DEBUG)



class transpositiontable:
	"""
	Hash table inspired by Zulko's easyAI: https://github.com/Zulko/easyAI/blob/master/easyAI/AI/DictTT.py
	"""

	def __init__(self, num_buckets=1024):
		self.dict = []
		for i in range(int(num_buckets)):
			self.dict.append((None, None))
		# logging.info('Transposition table size (MB): {}'.format(sys.getsizeof(self.tt.dict)/1e6))
		self.num_collisions = 0

	def hash_key(self, key):
		"""
		Given a key this will create a number and convert it to an index for the dict.
		"""
		return hash(key) % len(self.dict)

	def get(self, key):
		i, k, v = self.get_slot(key)
		return v

	def get_slot(self, key):
		slot = self.hash_key(key)

		if key == self.dict[slot][0]: # check this entry is the one we're looking for
			return slot, self.dict[slot][0], self.dict[slot][1]
		else:
			return -1, key, None

	def set(self, key, value):
		slot = self.hash_key(key)

		if self.dict[slot] != (None, None):
			self.num_collisions += 1

		self.dict[slot] = (key, value)

	def delete(self, key):
		slot = self.hash_key(key)
		self.dict[slot] = (None, None)


class Bot(object):

	settings = dict()
	round = -1
	board = np.zeros((6, 7), dtype=np.uint8)  # Access with [row_nr, col_nr]. [0,0] is on the top left.
	timeout = -1

	def make_turn(self):
		""" This method is for calculating and executing the next play.
			Make the play by calling place_disc exactly once.
		"""
		raise NotImplementedError()

	def place_disc(self, column):
		""" Writes your next play in stdout. """
		stdout.write("place_disc %d\n" % column)
		stdout.flush()

	def get_available_moves(self, board):
		""" Returns list of valid column numbers where a disc can be placed.
			(i.e. columns which haven't been filled to the top) """
		return np.where(board[0,:]==0)[0]

	def simulate_place_disc(self, board, col_nr, curr_player):
		""" Returns a board state after curr_player placed a disc in col_nr.
			This is a simulation and doesn't update the actual playing board. """
		# stopwatch = time.time()
		if board[0, col_nr] != 0:
			raise Bot.ColumnFullException()
		new_board = np.copy(board)
		# for row_nr in reversed(range(self.rows())):
		# 	if new_board[row_nr, col_nr] == 0:
		# 		new_board[row_nr, col_nr] = curr_player
		# 		return new_board
		new_board[np.max(np.where(new_board[:,col_nr]==0)),col_nr] = curr_player
		# self.simulate_move_times.append(time.time() - stopwatch)
		# if len(self.simulate_move_times) >= 3:
		# 	# logging.inf('Mean simulate move time +- SD: {} +- {}'.format(np.mean(self.simulate_move_times), np.std(self.simulate_move_times)))
		return new_board

	def me(self):
		""" Returns own bot id. """
		return self.settings['your_botid']

	def him(self):
		""" Returns opponent bot id. """
		return self.settings['opponent_botid']

	def rows(self):
		""" Returns amount of rows. """
		return self.settings['field_rows']

	def cols(self):
		""" Returns amount of columns. """
		return self.settings['field_columns']

	def current_milli_time(self):
		""" Returns current system time in milliseconds. """
		return int(round(time.time() * 1000))

	def set_timeout(self, millis):
		""" Sets time left until timeout in milliseconds. """
		self.timeout = self.current_milli_time() + millis

	def time_left(self):
		""" Get how much time is left until a timeout. """
		return self.timeout - self.current_milli_time()

	def run(self):
		""" Main loop.
		"""
		while not stdin.closed:
			try:
				rawline = stdin.readline()

				# End of file check
				if len(rawline) == 0:
					break

				line = rawline.strip()

				# Empty lines can be ignored
				if len(line) == 0:
					continue

				parts = line.split()

				command = parts[0]

				self.parse_command(command, parts[1:])

			except EOFError:
				return

	def parse_command(self, command, args):
		if command == 'settings':
			key, value = args
			if key in ('timebank', 'time_per_move', 'field_columns', 'field_rows'):
				value = int(value)
			self.settings[key] = value
			if key == 'your_botid':
				self.settings['your_botid'] = int(value)
				# logging.info('My botid: {}'.format(self.me()))
				if self.me() == 1:
					self.settings['opponent_botid'] = 2
				elif self.me() == 2:
					self.settings['opponent_botid'] = 1
				# logging.info('Opponent botid: {}'.format(self.him()))

		elif command == 'update':
			sub_command = args[1]
			args = args[2:]

			if sub_command == 'round':
				self.round = int(args[0])
			elif sub_command == 'field':
				self.parse_field(args[0])

		elif command == 'action':
			self.set_timeout(int(args[1]))
			self.make_turn()

	def parse_field(self, str_field):
		self.board = np.fromstring(str_field.replace(';', ','), sep=',', dtype=np.uint8).reshape(self.rows(), self.cols())

	class ColumnFullException(Exception):
		""" Raised when attempting to place disk in full column. """


class JDbot(Bot):

	simulate_move_times = []
	evaluate_times = []
	round_timeout = -1
	num_evals = 0 # number of board evaluations done
	tt = transpositiontable(num_buckets=1e7)

	def get_rounds_left(self):
		""" Returns estimated number of rounds left in game. """
		estimated_number_of_turns_in_a_match = 35
		return estimated_number_of_turns_in_a_match - self.round

	def time_for_round(self):
		""" Sets estimated optimal time to spend on this turn.
			Always keeps a minimum of 100ms in the timebank. """
		return min(self.time_left() / max(self.get_rounds_left(),1) + 500, self.time_left() - 100)

	def set_round_timeout(self):
		""" Sets time left until timeout in milliseconds. """
		self.round_timeout = self.current_milli_time() + self.time_for_round()

	def round_time_left(self):
		""" Get how much time is left to optimally spend this round. """
		return self.round_timeout - self.current_milli_time()

	def make_turn(self):
		# logging.info('\n-------------------------------- next turn -------------------------------------')
		# logging.info('\n{}'.format(self.board))

		# logging.info('Time left: {}'.format(self.time_left()))
		# logging.info('Round: {}'.format(self.round))

		stopwatch = time.time()
		move = self.alphabeta(self.board)
		# logging.info('Choosing a move took: {} milliseconds'.format((time.time() - stopwatch) * 1000))

		# logging.info('Chosen move: {}'.format(move))
		
		new_board = self.simulate_place_disc(self.board, move, self.me())
		# logging.info('\n{}'.format(new_board))

		board_value = self.evaluate_board(new_board)
		# logging.info('Board value: {}'.format(board_value))

		# logging.info('Board evaluations this turn: {}'.format(self.num_evals))
		self.num_evals = 0

		self.place_disc(move)

	def alphabeta(self, board, max_depth=float('inf'), min_time_left = 5000, min_depth=3):

		def max_value(board, alpha, beta, depth):
			if self.round_time_left() <= 0:
				return self.evaluate_board(board)
			if depth >= run_depth:
				return self.evaluate_board(board)
			if self.winning_board(board, self.me()):
				return float('inf')
			if self.winning_board(board, self.him()):
				return -float('inf')
			v = -float('inf')
			for new_board in map(lambda move: self.simulate_place_disc(board, move, self.me()), self.get_available_moves(board)):
				v = max(v, min_value(new_board, alpha, beta, depth + 1))
				if v >= beta:
					return v
				alpha = max(alpha, v)
			return v

		def min_value(board, alpha, beta, depth):
			if self.round_time_left() <= 0:
				return self.evaluate_board(board)
			if depth >= run_depth:
				return self.evaluate_board(board)
			if self.winning_board(board, self.me()):
				return float('inf')
			if self.winning_board(board, self.him()):
				return -float('inf')
			v = float('inf')
			for new_board in map(lambda move: self.simulate_place_disc(board, move, self.him()), self.get_available_moves(board)):
				v = min(v, max_value(new_board, alpha, beta, depth + 1))
				if v <= alpha:
					return v
				beta = min(beta, v)
			return v

		# sets estimated optimal time to spend on this round
		self.set_round_timeout()
		# logging.info('Estimated time to spend on this round: {}ms'.format(self.round_time_left()))

		values = np.array([-float('inf')]*7)
		alpha = -float('inf')
		beta = float('inf')

		# iterative deepening
		run_depth = 1
		while run_depth <= max_depth:

			new_values = values.copy()
			for m in self.get_available_moves(board):
				# explore tree up to set depth
				v = min_value(self.simulate_place_disc(board, m, self.me()), alpha, beta, 0)
				new_values[m] = v
				if self.round_time_left() <= 0:
					break

			# logging.info('Move values: {}'.format(new_values))

			if np.max(new_values) == -float('inf'):
				# logging.info("I think we're going to lose, but I'm going to delay it for as long as I can.")
				break

			values = new_values

			if np.count_nonzero(values!=-float('inf'))==1:
				# only one option, no point in exploring the tree further
				break

			if np.any(new_values==float('inf')):
				# logging.info("I think we're going to win!")
				break

			if self.round_time_left() <= 0:
				# logging.info('Reached depth of: {}'.format(run_depth))
				break

			run_depth += 1

		# slightly prefer certain columns
		values += np.array([0,1,2,3,2,1,0])
		
		# logging.info('Move values: {}'.format(values))

		if np.max(values) == -float('inf'):
			# all moves lead to losing. pick a valid move at random.
			moves = self.get_available_moves(board)
			return moves[np.random.randint(0,len(moves))]

		move = np.argmax(values)
		return move

	def winning_board(self, board, player):
		for row in board:
			if np.count_nonzero(row==player) < 4:
				continue
			for hit in KnuthMorrisPratt(row, self.patterns[player]['win']):
				return True
		for col in board.T:
			if np.count_nonzero(col==player) < 4:
				continue
			for hit in KnuthMorrisPratt(col, self.patterns[player]['win']):
				return True
		# top left to bottom right diagonals
		for i in range(-2,3):
			diag = board.diagonal(i)
			if np.count_nonzero(diag==player) < 4:
				continue
			for hit in KnuthMorrisPratt(diag, self.patterns[player]['win']):
				return True
		# top right to bottom left diagonals
		reversed_columns = board[:,::-1]
		for i in range(-2,3):
			diag = reversed_columns.diagonal(i)
			if np.count_nonzero(diag==player) < 4:
				continue
			for hit in KnuthMorrisPratt(diag, self.patterns[player]['win']):
				return True
		return False


	def evaluate_vector(self, vector, player, patterns):
		if np.count_nonzero(vector==player) < 3:
			return []
		# value = 0
		threats = []
		for pattern in patterns:
			for hit in KnuthMorrisPratt(vector, pattern):
				# find threat position
				threats.append(np.where(pattern==0)[0][0] + hit)
				# value += 1
		return threats

	def evaluate_board(self, board):

		self.num_evals += 1

		# stopwatch = time.time()

		# row_and_diag_patterns = np.array([[1,0,1,1], [1,1,0,1], [0,1,1,1], [1,1,1,0]])
		# col_patterns = np.array([[0,1,1,1]])
		row_multiplier = 10
		col_mutliplier = 10
		diag_multiplier = 10
		even_odd_multiplier = 1.5

		# first check if this board is in our hash table
		value = self.tt.get(str(board))
		if value != None:
			return value

		value = 0

		# check if we've won or lost
		if self.winning_board(board, self.me()):
			return float('inf')
		if self.winning_board(board, self.him()):
			return -float('inf')

		# check if board contains favorable situations for either player

		for player in [1,2]:

			v = 0

			# evaluate rows
			for i,row in enumerate(board[::-1]):
				i += 1 # one-based indexing for correct even/odd calculation
				threats = self.evaluate_vector(row, player, self.patterns[player]['row_and_diag'])
				# player 1 wants odd threats, player 2 wants even threats
				if (player==1 and not i%2==0) or (player==2 and i%2==0):
					v += len(threats) * even_odd_multiplier * row_multiplier
				else:
					v += len(threats) * row_multiplier

			# evaluate columns
			for col in board[::-1].T:
				threats = self.evaluate_vector(col, player, self.patterns[player]['col'])
				for t in threats:
					t += 1 # for even/odd calculation
					# player 1 wants odd threats, player 2 wants even threats
					if (player==1 and not t%2==0) or (player==2 and t%2==0):
						v += even_odd_multiplier * col_mutliplier
					else:
						v += col_mutliplier

			# evaluate diagonals
			reversed_columns = board[:,::-1]
			for i in range(-2,4):
				# top left to bottom right
				threats = self.evaluate_vector(board.diagonal(i)[::-1], player, self.patterns[player]['row_and_diag'])
				# top right to bottom left
				threats += self.evaluate_vector(reversed_columns.diagonal(i)[::-1], player, self.patterns[player]['row_and_diag'])
				for t in threats:
					# calculate row index of threat
					if i >= 2:
						t += i-1
					t += 1

					if (player==1 and not t%2==0) or (player==2 and t%2==0):
						v += even_odd_multiplier * col_mutliplier
					else:
						v += col_mutliplier

			if player == self.me():
				value += v
			else:
				value -= v

			# store value in transposition table
			self.tt.set(str(board), value)

		return value

	patterns = {1: {
					'win': np.array([1,1,1,1]),
					'row_and_diag': np.array([[1,0,1,1], [1,1,0,1], [0,1,1,1], [1,1,1,0]]),
					'col': np.array([[1,1,1,0]])
				},
				2: {
					'win': np.array([2,2,2,2]),
					'row_and_diag': np.array([[2,0,2,2], [2,2,0,2], [0,2,2,2], [2,2,2,0]]),
					'col': np.array([[2,2,2,0]])
				}
			}

	def test(self):
		self.settings['your_botid'] = 1
		self.settings['opponent_botid'] = 2
		board = np.array([	[1,0,2,1,0,0,0],
							[2,0,1,2,2,0,0],
							[1,0,2,2,1,2,0],
							[2,0,1,1,2,1,2],
							[2,0,2,2,1,1,2],
							[1,1,1,2,1,2,1]])
		print self.evaluate_board(board)
		# board = np.array([	[0,0,0,0,0,0,0],
		# 					[0,0,0,0,1,0,0],
		# 					[0,0,0,0,2,2,0],
		# 					[0,0,1,1,1,2,2],
		# 					[1,1,2,2,1,1,2],
		# 					[2,1,1,2,1,2,1]])
		# print self.evaluate_board(board)
		# board = np.array([	[0,0,0,0,0,0,0],
		# 					[0,0,0,0,0,0,0],
		# 					[0,0,0,1,0,0,0],
		# 					[0,0,2,1,0,0,0],
		# 					[0,0,2,2,0,0,0],
		# 					[0,1,2,1,0,0,0]])
		# print self.evaluate_board(board)
		# board = np.array([	[0,0,1,2,0,0,0],
		# 					[0,0,2,2,0,0,0],
		# 					[0,0,2,1,0,0,0],
		# 					[0,2,1,1,0,0,0],
		# 					[0,1,1,2,0,0,0],
		# 					[1,1,2,1,0,0,0]])
		# print self.evaluate_board(board)

if __name__ == '__main__':
	""" Run the bot! """

	# try:
	JDbot().run()
		# StarterBot().test()
	# except:
		# logging.exception("Oops:")
