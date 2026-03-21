%define the chess board
rows(1..8).
cols(1..8).
square(R, C):- rows(R), cols(C).

%column display names
%col_name(1,a). col_name(2,b). col_name(3,c). col_name(4,d).
%col_name(5,e). col_name(6,f). col_name(7,g). col_name(8,h).

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

%movement states - we need to keep track of the board state before and after the move, as pieces can be captured and removed from the board, which affects legal moves and threats.
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


%genetating the possible moves for white - we want to find a move that puts black in checkmate, 
%so we assume black will play optimally and move their king out of check if possible. 
%therefore we only need to generate one move for white, as if there is a winning move, 
%it will be the same regardless of which move white chooses.
%we only want 1 move for white, the piece must be placed on the board, and the move must be legal
1{ 
    move(ID, R1, C1, R2, C2): 
                            legal_move(ID, R1, C1, R2, C2), 
                            placement(white, _, ID, R1, C1)
} 1.


%now, defining the board state after the move 
%after the move, the piece will be in the new position, and the old position will be empty.
%we can define the new placement of the piece after the move
new_placement(ID, R2, C2):- 
                            move(ID, _, _, R2, C2).%after the move, the moved piece is at its new square.
new_placement(ID, R, C):- 
                            placement(_, _, ID, R, C), 
                            not move(ID, _, _, _, _), 
                            not captured(_, ID, _, _). %if the piece is not moved, it stays in the same place.

%defining an attack
%See if piece was captured - check if after a move, if there is another piece in the new position, then the piece in the new position was captured.
captured(ID1, ID2, R2, C2):- 
                            new_placement(ID1, R2, C2), 
                            placement(_, _, ID2, R2, C2), ID1!=ID2. 

%a piece moves to a square, lets say white.
%a square is attacked by white if the piece, in the posn after it moved, could legally move to that square 
%we need the new posn - to get the current posn, and then from this posn, track squareswe could move into
threaten(ID, R1, C1, R2, C2):- 
                            new_placement(ID, R1, C1), 
                            legal_queen_move(after, R1, C1, R2, C2),
                            placement(white, queen, ID, _, _).

threaten(ID, R1, C1, R2, C2):-
                            new_placement(ID, R1, C1), 
                            legal_king_move(white, R1, C1, R2, C2),
                            placement(white, king, ID, _, _).



%check if black king is in check
in_check(black):- 
                    threaten(_, _, _, R, C), 
                    placement(black, king, bk, R, C).

%check if black king can escape
can_escape(black):- 
                    new_placement(bk, R1, C1), 
                    legal_king_move(black, R1, C1, R2, C2), 
                    not threaten(_, _, _, R2, C2).


%checkmate - black is in check and cannot escape
checkmate(black):- in_check(black), not can_escape(black).

%only keep sets where checkmate is true
:- not checkmate(black).

#show move/5.
#show in_check/1.
#show can_escape/1.
#show checkmate/1.