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
% Extended rules to find mate-in-one

% Choose a white move
{ chosen_move(ID, R1, C1, R2, C2) } :- legal_move(ID, R1, C1, R2, C2), to_move(white), placement(white, _, ID, R1, C1).

% Exactly one move chosen
:- #count{ ID, R1, C1, R2, C2 : chosen_move(ID, R1, C1, R2, C2) } != 1.

% Compute new placements after the chosen white move
% The moved piece goes to new square
new_placement(ID, R2, C2) :- chosen_move(ID, _, _, R2, C2).
% Other white pieces stay
new_placement(ID, R, C) :- placement(white, _, ID, R, C), not chosen_move(ID, R, C, _, _).
% Black king stays (unless captured, but king capture shouldn't happen in legal chess)
new_placement(bk, R, C) :- placement(black, king, bk, R, C).

% After white moves, determine new positions
% White queen new position
white_queen_after(R2, C2) :- chosen_move(wq, _, _, R2, C2).
white_queen_after(R, C) :- placement(white, queen, wq, R, C), not chosen_move(wq, R, C, _, _).

% White king position (unchanged since we only move one piece)
white_king_after(R, C) :- placement(white, king, wk, R, C), not chosen_move(wk, R, C, _, _).
white_king_after(R2, C2) :- chosen_move(wk, _, _, R2, C2).

% Black king position after white move
black_king_after(R, C) :- placement(black, king, bk, R, C).

% Squares attacked by white queen after the move
% Need to check if queen's attacks are blocked by pieces in the after state

% Pieces on board after white move (for blocking purposes)
piece_after(R, C) :- new_placement(_, R, C).

% Queen attacks a square if it can reach it without being blocked
queen_attacks_after(QR, QC, TR, TC) :-
    white_queen_after(QR, QC),
    queen_moves(QR, QC, TR, TC),
    not queen_blocked_after(QR, QC, TR, TC).

% Queen is blocked after move
queen_blocked_after(QR, QC, TR, TC) :-
    white_queen_after(QR, QC),
    between_horizontal(QR, QC, BC, TC),
    piece_after(QR, BC), TC = TR, QR = TR.

queen_blocked_after(QR, QC, TR, TC) :-
    white_queen_after(QR, QC),
    between_horizontal(QR, TC, BC, QC),
    piece_after(QR, BC), TC = TR, QR = TR.

queen_blocked_after(QR, QC, TR, TC) :-
    white_queen_after(QR, QC),
    between_vertical(QC, QR, BR, TR),
    piece_after(BR, QC), TC = QC.

queen_blocked_after(QR, QC, TR, TC) :-
    white_queen_after(QR, QC),
    between_vertical(QC, TR, BR, QR),
    piece_after(BR, QC), TC = QC.

queen_blocked_after(QR, QC, TR, TC) :-
    white_queen_after(QR, QC),
    between_diagonal(BR, BC, QR, QC, TR, TC),
    piece_after(BR, BC).

% Squares attacked by white king after the move
king_attacks_after(KR, KC, TR, TC) :-
    white_king_after(KR, KC),
    king_moves(KR, KC, TR, TC).

% A square is safe for black king if not attacked by white queen or white king
square_safe_for_black(R, C) :-
    square(R, C),
    not queen_attacks_after(_, _, R, C),
    not king_attacks_after(_, _, R, C).

% Black king is in check after white move
black_in_check :-
    black_king_after(BKR, BKC),
    queen_attacks_after(_, _, BKR, BKC).

black_in_check :-
    black_king_after(BKR, BKC),
    king_attacks_after(_, _, BKR, BKC).

% Black king can escape to a square if:
% 1. It's a legal king move
% 2. The square is not occupied by a black piece
% 3. The square is safe (not attacked by white)
% 4. If it's occupied by a white piece, black can capture it (but then it must be undefended)

% Squares occupied by white after move
white_piece_after(R, C) :- white_queen_after(R, C).
white_piece_after(R, C) :- white_king_after(R, C).

% Black king escape squares
black_can_escape :-
    black_king_after(BKR, BKC),
    king_moves(BKR, BKC, TR, TC),
    not white_piece_after(TR, TC),  % not occupied by white (can't capture defended piece)
    square_safe_for_black(TR, TC).

% Black can capture white queen if queen is not defended
% Queen is defended by white king if white king attacks queen's square
queen_defended :-
    white_queen_after(QR, QC),
    king_attacks_after(_, _, QR, QC).

% Black king can capture white queen if queen is not defended
black_can_capture_queen :-
    black_king_after(BKR, BKC),
    white_queen_after(QR, QC),
    king_moves(BKR, BKC, QR, QC),
    not queen_defended,
    % After capturing queen, black king must not be in check from white king
    not king_attacks_after(_, _, QR, QC).

% It's checkmate if black is in check and cannot escape and cannot capture
checkmate :-
    black_in_check,
    not black_can_escape,
    not black_can_capture_queen.

% The chosen move must result in checkmate
:- not checkmate.

% Output the mating move
mate_move(white, Type, R1, C1, R2, C2) :-
    chosen_move(ID, R1, C1, R2, C2),
    placement(white, Type, ID, R1, C1).

#show mate_move/6.