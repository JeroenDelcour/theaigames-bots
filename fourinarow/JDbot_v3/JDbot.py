# Python bot for the four-in-a-row AI challenge at www.theaigames.com
# by Jeroen Delcour <jeroendecour@gmail.com>
#
# Built upon Lukas Knoepfel's <shylux@gmail.com> starter bot (version 1.0, 30 Mar 2016)
# under the MIT license (http://opensource.org/licenses/MIT)

from sys import stdin, stdout
import numpy as np
from KnuthMorrisPratt import KnuthMorrisPratt
import time
import logging
logging.basicConfig(filename='JDbot2.log',format='%(asctime)s %(levelname)s: %(message)s',level=logging.DEBUG)

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
		# 	logging.inf('Mean simulate move time +- SD: {} +- {}'.format(np.mean(self.simulate_move_times), np.std(self.simulate_move_times)))
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
				logging.info('My botid: {}'.format(self.me()))
				if self.me() == 1:
					self.settings['opponent_botid'] = 2
				elif self.me() == 2:
					self.settings['opponent_botid'] = 1
				logging.info('Opponent botid: {}'.format(self.him()))

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


class StarterBot(Bot):

	simulate_move_times = []
	evaluate_times = []
	round_timeout = -1

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
		logging.info('\n-------------------------------- next turn -------------------------------------')
		logging.info('\n{}'.format(self.board))

		logging.info('Time left: {}'.format(self.time_left()))
		logging.info('Round: {}'.format(self.round))

		stopwatch = time.time()
		move = self.alphabeta(self.board)
		logging.info('Choosing a move took: {} milliseconds'.format((time.time() - stopwatch) * 1000))

		logging.info('Chosen move: {}'.format(move))
		
		new_board = self.simulate_place_disc(self.board, move, self.me())
		logging.info('\n{}'.format(new_board))

		board_value = self.evaluate_board(new_board)
		logging.info('Board value: {}'.format(board_value))

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
		logging.info('Estimated time to spend on this round: {}ms'.format(self.round_time_left()))

		values = np.array([-float('inf')]*7)
		alpha = -float('inf')
		beta = float('inf')

		# iterative deepening
		run_depth = 1
		while run_depth <= max_depth:
			for m in self.get_available_moves(board):
				v = min_value(self.simulate_place_disc(board, m, self.me()), alpha, beta, 0)
				if self.round_time_left() > 0:
					values[m] = v
				else:
					break

			logging.info('Move values: {}'.format(values))

			if np.max(values) == -float('inf'):
				logging.info("End of game determined. We've lost.")
				# every move leads to losing, so just pick one at random
				moves = self.get_available_moves(board)
				return moves[np.random.randint(0,len(moves))]

			if np.any(values==float('inf')):
				logging.info("End of game determined. We've won!")
				break

			if self.round_time_left() <= 0:
				logging.info('Reached depth of: {} with {}ms time left'.format(run_depth, self.round_time_left()))
				break

			run_depth += 1

		values += np.array([1,0,0,2,0,0,1])
		logging.info('Move values: {}'.format(values))

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
			return 0
		value = 0
		for pattern in patterns:
			for hit in KnuthMorrisPratt(vector, pattern):
				value += 1
		return value

	def evaluate_board(self, board):

		# stopwatch = time.time()

		# row_and_diag_patterns = np.array([[1,0,1,1], [1,1,0,1], [0,1,1,1], [1,1,1,0]])
		# col_patterns = np.array([[0,1,1,1]])
		row_multiplier = 10
		col_mutliplier = 10
		diag_multiplier = 10

		value = 0

		# first check if we've won or lost
		if self.winning_board(board, self.me()):
			return float('inf')
		if self.winning_board(board, self.him()):
			return -float('inf')

		# check if board contains favorable situations for either player
		# evaluate rows
		for row in board:
			value += self.evaluate_vector(row, self.me(), self.patterns[self.me()]['row_and_diag']) * row_multiplier
			value -= self.evaluate_vector(row, self.him(), self.patterns[self.him()]['row_and_diag']) * row_multiplier
		# for r in board:
		# 	value += evaluate_row(r, row_and_diag_patterns*self.me()) * row_multiplier
		# 	value -= evaluate_row(r, row_and_diag_patterns*self.him()) * row_multiplier
		# evaluate columns
		for col in board.T:
			value += self.evaluate_vector(col, self.me(), self.patterns[self.me()]['col']) * col_mutliplier
			value -= self.evaluate_vector(col, self.him(), self.patterns[self.him()]['col']) * col_mutliplier
		# evaluate diagonals
		reversed_columns = board[:,::-1]
		for i in range(-2,3):
			# top left to bottom right diagonals
			value += self.evaluate_vector(board.diagonal(i), self.me(), self.patterns[self.me()]['row_and_diag']) * diag_multiplier
			value -= self.evaluate_vector(board.diagonal(i), self.him(), self.patterns[self.him()]['row_and_diag']) * diag_multiplier
			# reverse columns to get the top right to bottom left diagonals
			value += self.evaluate_vector(reversed_columns.diagonal(i), self.me(), self.patterns[self.me()]['row_and_diag']) * diag_multiplier
			value -= self.evaluate_vector(reversed_columns.diagonal(i), self.him(), self.patterns[self.him()]['row_and_diag']) * diag_multiplier

		# self.evaluate_times.append(time.time() - stopwatch)

		return value

	patterns = {1: {
					'win': np.array([1,1,1,1]),
					'row_and_diag': np.array([[1,0,1,1], [1,1,0,1], [0,1,1,1], [1,1,1,0]]),
					'col': np.array([[0,1,1,1]])
				},
				2: {
					'win': np.array([2,2,2,2]),
					'row_and_diag': np.array([[2,0,2,2], [2,2,0,2], [0,2,2,2], [2,2,2,0]]),
					'col': np.array([[0,2,2,2]])
				}
			}

	def test(self):
		self.settings['your_botid'] = 1
		self.settings['opponent_botid'] = 2
		board = np.array([	[0,0,0,0,0,0,0],
							[0,0,0,0,0,0,0],
							[0,0,2,1,0,0,0],
							[0,0,2,1,0,0,0],
							[0,0,2,2,0,0,0],
							[0,1,2,1,0,0,0]])
		print self.evaluate_board(board)
		board = np.array([	[0,0,0,0,0,0,0],
							[0,0,0,0,0,0,0],
							[0,0,0,1,0,0,0],
							[0,0,2,1,0,0,0],
							[0,0,2,2,0,0,0],
							[0,1,2,1,0,0,0]])
		print self.evaluate_board(board)
		board = np.array([	[0,0,1,2,0,0,0],
							[0,0,2,2,0,0,0],
							[0,0,2,1,0,0,0],
							[0,2,1,1,0,0,0],
							[0,1,1,2,0,0,0],
							[1,1,2,1,0,0,0]])
		print self.evaluate_board(board)

if __name__ == '__main__':
	""" Run the bot! """

	try:
		StarterBot().run()
		# StarterBot().test()
	except:
		logging.exception("Oops:")
