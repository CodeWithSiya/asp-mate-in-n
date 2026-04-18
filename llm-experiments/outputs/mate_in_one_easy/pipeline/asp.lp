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
horizontal(R, C1, R, C2):- square(R, C1), square(R, C2), C1!=C2.
vertical(R1, C, R2, C):-  square(R1, C), square(R2, C), R1!=R2.
diagonal(R1, C1, R2, C2):- square(R1, C1), square(R2, C2), |R1-R2| = |C1-C2|, (R1, C1) != (R2, C2).

queen_moves(R1, C1, R2, C2):- horizontal(R1, C1, R2, C2).
queen_moves(R1, C1, R2, C2):- vertical(R1, C1, R2, C2).
queen_moves(R1, C1, R2, C2):- diagonal(R1, C1, R2, C2).

king_moves(R1, C1, R2, C2):- square(R1, C1), square(R2, C2), |C1-C2| <= 1, |R1-R2| <=1, (R1, C1) != (R2, C2).

between_horizontal(R_curr, C_left, C_curr, C_right):- square(R_curr, C_curr), square(R_curr, C_left), square(R_curr, C_right), C_left<C_curr, C_curr<C_right.
between_vertical(C_curr, R_top, R_curr, R_bottom):- square(R_curr, C_curr), square(R_top, C_curr), square(R_bottom, C_curr), R_top<R_curr, R_curr<R_bottom.

between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- square(R_curr, C_curr), square(R1, C1), square(R2, C2), |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, R1<R_curr, R_curr<R2, C1<C_curr, C_curr<C2.
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- square(R_curr, C_curr), square(R1, C1), square(R2, C2), |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, R2<R_curr, R_curr<R1, C2<C_curr, C_curr<C1.
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- square(R_curr, C_curr), square(R1, C1), square(R2, C2), |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, R1<R_curr, R_curr<R2, C1>C_curr, C_curr>C2.
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- square(R_curr, C_curr), square(R1, C1), square(R2, C2), |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, R2<R_curr, R_curr<R1, C2>C_curr, C_curr>C1.

state(before; after).

occupied(before, R, C):- placement(_, _, _, R, C).
occupied(after, R, C):- new_placement(_, R, C).

friendly(P, R, C):- placement(P, _, _, R, C).

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

black_threatens(R2, C2):-
    placement(black, king, bk, R1, C1),
    king_moves(R1, C1, R2, C2).

white_threatens(R2, C2):-
    placement(white, king, wk, R1, C1),
    king_moves(R1, C1, R2, C2).

legal_queen_move(S, R1, C1, R2, C2):-
    state(S),
    queen_moves(R1, C1, R2, C2),
    not blocked(S, R1, C1, R2, C2),
    not friendly(white, R2, C2).

legal_king_move(P, R1, C1, R2, C2):-
    player(P),
    king_moves(R1, C1, R2, C2),
    not friendly(P, R2, C2).

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

% Choose exactly one white move
{ chosen_move(ID, R1, C1, R2, C2) : legal_move(ID, R1, C1, R2, C2), placement(white, _, ID, R1, C1) } = 1.

% New placements after white's chosen move
new_placement(ID, R2, C2) :- chosen_move(ID, _, _, R2, C2).
new_placement(ID, R, C) :- placement(white, _, ID, R, C), not chosen_move(ID, R, C, _, _).
new_placement(ID, R, C) :- placement(black, _, ID, R, C), not chosen_move(_, _, _, R, C).

% Piece positions after white's move
piece_after(Color, Type, ID, R, C) :-
    placement(Color, Type, ID, R, C),
    new_placement(ID, R, C).
piece_after(Color, Type, ID, R2, C2) :-
    placement(Color, Type, ID, R1, C1),
    chosen_move(ID, R1, C1, R2, C2).

% Occupied squares after white's move
occupied_after(R, C) :- new_placement(_, R, C).

% Blocking after white's move (used for checking black king is in check)
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

% White queen attacks after white's move
white_queen_attacks_after(R2, C2) :-
    piece_after(white, queen, _, R1, C1),
    queen_moves(R1, C1, R2, C2),
    not blocked_after(R1, C1, R2, C2).

% White king attacks after white's move
white_king_attacks_after(R2, C2) :-
    piece_after(white, king, _, R1, C1),
    king_moves(R1, C1, R2, C2).

white_attacks_after(R, C) :- white_queen_attacks_after(R, C).
white_attacks_after(R, C) :- white_king_attacks_after(R, C).

% Black king is in check after white's move
black_king_in_check_after :-
    piece_after(black, king, _, R, C),
    white_attacks_after(R, C).

% Occupied after white's move excluding the black king (for escape square computation)
occupied_after_no_bk(R, C) :-
    new_placement(ID, R, C),
    not placement(black, king, ID, R, C).

% Blocking without black king present
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

% White queen attacks without black king blocking
white_queen_attacks_no_bk(R2, C2) :-
    piece_after(white, queen, _, R1, C1),
    queen_moves(R1, C1, R2, C2),
    not blocked_after_no_bk(R1, C1, R2, C2).

% White king attacks (unchanged by black king position)
white_king_attacks_no_bk(R2, C2) :-
    piece_after(white, king, _, R1, C1),
    king_moves(R1, C1, R2, C2).

white_attacks_no_bk(R, C) :- white_queen_attacks_no_bk(R, C).
white_attacks_no_bk(R, C) :- white_king_attacks_no_bk(R, C).

% Black pieces after white's move (for friendly piece check)
black_piece_after(R, C) :- piece_after(black, _, _, R, C).

% Black king can escape to (R2,C2) if:
% 1. Valid king move from current black king position
% 2. Not occupied by a black piece at destination
% 3. Not attacked by white (with black king removed from original square)
% 4. If destination has a white piece, that piece must not be protected (already handled by white_attacks_no_bk)
black_king_can_escape :-
    piece_after(black, king, _, BKR, BKC),
    king_moves(BKR, BKC, R2, C2),
    not black_piece_after(R2, C2),
    not white_attacks_no_bk(R2, C2).

% Checkmate: black king in check and cannot escape
checkmate_after :-
    black_king_in_check_after,
    not black_king_can_escape.

% Constraint: chosen move must result in checkmate
:- not checkmate_after.

% Output the mating move
mate_move(white, Type, C1, R1, C2, R2) :-
    chosen_move(ID, R1, C1, R2, C2),
    placement(white, Type, ID, R1, C1).

#show mate_move/6.