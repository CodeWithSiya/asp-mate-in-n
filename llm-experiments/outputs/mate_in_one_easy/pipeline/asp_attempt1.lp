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

%---------------------------------------------------------------------------------------------------------------------------
% MATE-IN-ONE SOLVER EXTENSION

% Choose one white move to make
{ chosen_move(ID, R1, C1, R2, C2) : legal_move(ID, R1, C1, R2, C2), placement(white, _, ID, R1, C1) } = 1.

% Compute new placements after the chosen white move
% The moved piece goes to new square
new_placement(ID, R2, C2) :- chosen_move(ID, _, _, R2, C2).
% All other white pieces stay
new_placement(ID, R, C) :- placement(white, _, ID, R, C), not chosen_move(ID, R, _, _, _).
% Black pieces stay unless captured
new_placement(ID, R, C) :- placement(black, _, ID, R, C), not chosen_move(_, _, _, R, C).

% After the move, what piece is at each square
piece_after(Color, Type, ID, R, C) :-
    placement(Color, Type, ID, R, C),
    new_placement(ID, R, C).
piece_after(Color, Type, ID, R2, C2) :-
    placement(Color, Type, ID, R1, C1),
    chosen_move(ID, R1, C1, R2, C2).

% Occupied after the move
occupied_after(R, C) :- new_placement(_, R, C).

% Friendly pieces after move (for black king move legality)
friendly_after(Color, R, C) :- piece_after(Color, _, _, R, C).

% Blocking after the move
blocked_after(R1, C1, R2, C2) :-
    between_horizontal(R1, C1, C_curr, C2),
    occupied_after(R1, C_curr), R2 = R1.
blocked_after(R1, C1, R2, C2) :-
    between_horizontal(R1, C2, C_curr, C1),
    occupied_after(R1, C_curr), R2 = R1.
blocked_after(R1, C1, R2, C2) :-
    between_vertical(C1, R1, R_curr, R2),
    occupied_after(R_curr, C1), C2 = C1.
blocked_after(R1, C1, R2, C2) :-
    between_vertical(C1, R2, R_curr, R1),
    occupied_after(R_curr, C1), C2 = C1.
blocked_after(R1, C1, R2, C2) :-
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2),
    occupied_after(R_curr, C_curr).

% White queen attacks a square after the move (for checking black king)
white_queen_attacks_after(R2, C2) :-
    piece_after(white, queen, _, R1, C1),
    queen_moves(R1, C1, R2, C2),
    not blocked_after(R1, C1, R2, C2).

% White king attacks a square after the move
white_king_attacks_after(R2, C2) :-
    piece_after(white, king, _, R1, C1),
    king_moves(R1, C1, R2, C2).

% White attacks a square after the move
white_attacks_after(R, C) :- white_queen_attacks_after(R, C).
white_attacks_after(R, C) :- white_king_attacks_after(R, C).

% Black king is in check after white's move
black_king_in_check_after :-
    piece_after(black, king, _, R, C),
    white_attacks_after(R, C).

% Legal moves for black king after white's move
% Black king can move to a square that:
% 1. Is a valid king move
% 2. Is not occupied by a black piece
% 3. Is not attacked by white after the move (considering black king has moved)

% After black king moves to (R2,C2), white pieces are still at their after-move positions
% We need to check if (R2,C2) is attacked by white pieces (queen position unchanged by black's move)

% White queen attacks after black king moves to (R2,C2)
% The queen's position doesn't change; blocking may change because black king moved
% We need to recompute blocking without the black king at its original position but with it at new position

% Squares occupied after white move and after black king moves from (BKR1,BKC1) to (BKR2,BKC2)
occupied_after_bk_move(R, C, BKR1, BKC1, BKR2, BKC2) :-
    new_placement(ID, R, C),
    placement(black, king, ID, BKR1, BKC1),
    R = BKR2, C = BKC2.
occupied_after_bk_move(R, C, BKR1, BKC1, BKR2, BKC2) :-
    new_placement(ID, R, C),
    not (placement(black, king, ID, BKR1, BKC1), R = BKR1, C = BKC1),
    square(BKR1, BKC1), square(BKR2, BKC2).

% This approach is getting complex; let's simplify by checking if white attacks (R2,C2)
% ignoring the black king's original square (since it moved away)

% White queen attacks (R2,C2) after white move, with black king removed from (BKR1,BKC1)
% We check blocking without the black king at BKR1,BKC1

% Pieces on board after white move excluding black king
occupied_after_no_bk(R, C) :-
    new_placement(ID, R, C),
    not placement(black, king, ID, R, C).

blocked_after_no_bk(R1, C1, R2, C2) :-
    between_horizontal(R1, C1, C_curr, C2),
    occupied_after_no_bk(R1, C_curr), R2 = R1.
blocked_after_no_bk(R1, C1, R2, C2) :-
    between_horizontal(R1, C2, C_curr, C1),
    occupied_after_no_bk(R1, C_curr), R2 = R1.
blocked_after_no_bk(R1, C1, R2, C2) :-
    between_vertical(C1, R1, R_curr, R2),
    occupied_after_no_bk(R_curr, C1), C2 = C1.
blocked_after_no_bk(R1, C1, R2, C2) :-
    between_vertical(C1, R2, R_curr, R1),
    occupied_after_no_bk(R_curr, C1), C2 = C1.
blocked_after_no_bk(R1, C1, R2, C2) :-
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2),
    occupied_after_no_bk(R_curr, C_curr).

white_queen_attacks_after_no_bk(R2, C2) :-
    piece_after(white, queen, _, R1, C1),
    queen_moves(R1, C1, R2, C2),
    not blocked_after_no_bk(R1, C1, R2, C2).

white_king_attacks_after_no_bk(R2, C2) :-
    piece_after(white, king, _, R1, C1),
    king_moves(R1, C1, R2, C2).

white_attacks_after_no_bk(R, C) :- white_queen_attacks_after_no_bk(R, C).
white_attacks_after_no_bk(R, C) :- white_king_attacks_after_no_bk(R, C).

% Black king escape: can it move to (R2,C2)?
% Conditions:
% 1. king_moves from current position
% 2. Not occupied by black piece after white's move (can capture white piece if not protected)
% 3. Not attacked by white (with black king removed from original square)
% 4. If capturing a white piece, that square must not be protected by other white pieces

% After white move, black king is at (BKR, BKC)
% Black king tries to move to (R2, C2)

% Is (R2,C2) occupied by a white piece after white's move?
white_piece_at_after(R, C) :-
    piece_after(white, _, _, R, C).

% Black king can escape to (R2,C2) if:
% - It's a valid king move from black king's current position
% - Not occupied by a black piece
% - Not attacked by white (after removing black king from its square)
%   (if capturing white piece, white piece is removed too - but we approximate by checking attacks without bk)

black_king_can_escape :-
    piece_after(black, king, _, BKR, BKC),
    king_moves(BKR, BKC, R2, C2),
    not friendly_after(black, R2, C2),
    not white_attacks_after_no_bk(R2, C2).

% It's checkmate if black king is in check and cannot escape
checkmate_after :-
    black_king_in_check_after,
    not black_king_can_escape.

% The chosen move must result in checkmate
:- not checkmate_after.

% Output the mating move
mate_move(white, Type, C1, R1, C2, R2) :-
    chosen_move(ID, R1, C1, R2, C2),
    placement(white, Type, ID, R1, C1).

#show mate_move/6.