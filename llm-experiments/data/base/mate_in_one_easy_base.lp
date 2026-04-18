%define the chess board
rows(1..8).
cols(1..8).
square(R, C):- rows(R), cols(C).

%define players
player(white; black).

%define chess pieces
pieceType(king; queen).

%define the initial positions
%placement(player, piece, id, row, col)
placement(white, queen, wq, 7, 2).
placement(white, king, wk, 6, 8).
placement(black, king, bk, 8, 8).

%piece next to move
to_move(white).

%define the legal moves
horizontal(R, C1, R, C2):- square(R, C1), square(R, C2), C1!=C2. %horizontal movement
vertical(R1, C, R2, C):-  square(R1, C), square(R2, C), R1!=R2. %vertical movement
diagonal(R1, C1, R2, C2):- square(R1, C1), square(R2, C2), |R1-R2| = |C1-C2|, (R1, C1) != (R2, C2). %diagonal movement - absolute differences are equal, pairs arent equal

%defining legal moves
%Queen moves - hor, ver, diag - any no. of squares
queen_moves(R1, C1, R2, C2):- horizontal(R1, C1, R2, C2).
queen_moves(R1, C1, R2, C2):- vertical(R1, C1, R2, C2).
queen_moves(R1, C1, R2, C2):- diagonal(R1, C1, R2, C2).

%King moves - to adjacent squares in any direction
king_moves(R1, C1, R2, C2):- square(R1, C1),  square(R2, C2), |C1-C2| <= 1, |R1-R2| <=1, (R1, C1) != (R2, C2).

%---------------------------------------------------------------------------------------------------------------------------
%legal moves - not blocked by another piece, and not occupied by a friendly piece.

%define the pieces that the current piece is between
between_horizontal(R_curr, C_left, C_curr, C_right):- square(R_curr, C_curr), square(R_curr, C_left), square(R_curr, C_right), C_left<C_curr, C_curr<C_right.
between_vertical(C_curr, R_top, R_curr, R_bottom):- square(R_curr, C_curr), square(R_top, C_curr), square(R_bottom, C_curr), R_top<R_curr, R_curr<R_bottom.

between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- square(R_curr, C_curr), square(R1, C1), square(R2, C2), |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, R1<R_curr, R_curr<R2, C1<C_curr, C_curr<C2. %bottom left to top right diagonal
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- square(R_curr, C_curr), square(R1, C1), square(R2, C2), |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, R2<R_curr, R_curr<R1, C2<C_curr, C_curr<C1. %top right to bottom left diagonal
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- square(R_curr, C_curr), square(R1, C1), square(R2, C2), |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, R1<R_curr, R_curr<R2, C1>C_curr, C_curr>C2. %top left to bottom right diagonal
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- square(R_curr, C_curr), square(R1, C1), square(R2, C2), |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, R2<R_curr, R_curr<R1, C2>C_curr, C_curr>C1. %bottom right to top left diagonal

%movement states
state(before; after).

%defining occupied squares
occupied(before, R, C):- placement(_, _, _, R, C).
occupied(after, R, C):- new_placement(_, R, C).

%define friendly pieces - to stop a player from taking their own pieces
friendly(P, R, C):- placement(P, _, _, R, C).


%defining blocked moves - prevents moving through pieces
% eg - if theres a path from R1, C1 to R2, C2, and there is an occupied square in the way, then the move is blocked
blocked(S, R1, C1, R2, C2):- 
                            between_horizontal(R1, C1, C_curr, C2), 
                            occupied(S, R1, C_curr), R2 = R1.
blocked(S, R1, C1, R2, C2):- 
                            between_horizontal(R1, C2, C_curr, C1), 
                            occupied(S, R1, C_curr), R2 = R1.

blocked(S, R1, C1, R2, C2):- 
                            between_vertical(C1, R1, R_curr, R2), 
                            occupied(S, R_curr, C1), C2 = C1.
blocked(S, R1, C1, R2, C2):- 
                            between_vertical(C1, R2, R_curr, R1), 
                            occupied(S, R_curr, C1), C2 = C1.

blocked(S, R1, C1, R2, C2):- 
                            between_diagonal(R_curr, C_curr, R1, C1, R2, C2), 
                            occupied(S, R_curr, C_curr).


%squares the black king threatens
black_threatens(R2, C2):- 
                        placement(black, king, bk, R1, C1), %where black king is originally
                        king_moves(R1, C1, R2, C2).

%squares the white king threatens
white_threatens(R2, C2):- 
                        placement(white, king, wk, R1, C1),  %where white king is after move
                        king_moves(R1, C1, R2, C2).


%defining legal moves - not blocked. only modelling for white pieces as we want to find a winning move for white, so we assume black will play optimally and move their king out of check if possible.
legal_queen_move(S, R1, C1, R2, C2):- 
                                state(S),
                                queen_moves(R1, C1, R2, C2), %valid move for a queen
                                not blocked(S, R1, C1, R2, C2), % not blocked by another piece
                                not friendly(white, R2, C2). % not occupied by a friendly piece

legal_king_move(P, R1, C1, R2, C2):- 
                                player(P),  
                                king_moves(R1, C1, R2, C2), % valid move for a king
                                not friendly(P, R2, C2).% not occupied by another piece 


%piece must be placed on the board to move
legal_move(ID, R1, C1, R2, C2):- 
                                placement(white, queen, ID, R1, C1), 
                                legal_queen_move(before, R1, C1, R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
                                placement(white, king, ID, R1, C1), 
                                legal_king_move(white, R1, C1, R2, C2), 
                                not black_threatens(R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
                                placement(black, king, ID, R1, C1), 
                                legal_king_move(black, R1, C1, R2, C2), 
                                not white_threatens(R2, C2).
