Looking at the error, the issue is with `white_threatens_excl` rules where `ER` and `EC` appear only in conditions like `R1 != ER` without being bound elsewhere. I need to ensure `ER` and `EC` are properly grounded.

The fix is to bind `ER` and `EC` through `square(ER, EC)` in the head or through explicit grounding before using them in comparisons. I'll restructure the `white_threatens_excl` predicate to include `square(ER,EC)` as a body literal to ground those variables.

%define the chess board
rows(1..8).
cols(1..8).
square(R, C):- rows(R), cols(C).

%define players
player(white; black).

%define chess pieces
pieceType(king; queen; pawn; rook; bishop; knight).

%define the initial positions]
placement(white, king, wk, 8, 7).
placement(white, rook, wr1, 8, 4).
placement(white, pawn, wp1, 6, 8).
placement(white, pawn, wp2, 7, 7).
placement(white, pawn, wp3, 7, 6).

placement(black, pawn, bp1, 2, 6).
placement(black, pawn, bp2, 2, 7).
placement(black, pawn, bp3, 2, 8).
placement(black, pawn, bp4, 4, 1).
placement(black, king, bk, 1, 7).
placement(black, rook, br1, 7, 2).
placement(black, bishop, bb1, 5, 6).

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

%pawn moves - only forward, and can only capture diagonally
pawn_moves(white, R1, C1, R2, C2):- 
                                    square(R1, C1), square(R2, C2), 
                                    R2 = R1-1, C2 = C1, %forward move
                                    not occupied(before, R2, C2). %cannot move forward if occupied

pawn_moves(black, R1, C1, R2, C2):-
                                    square(R1, C1), square(R2, C2), 
                                    R2 = R1+1, C2=C1, %diagonal capture
                                    not occupied(before, R2, C2). %can only capture if occupied by an enemy piece
%white pawn diagonal capture
pawn_moves(white, R1, C1, R2, C2) :- 
                                    square(R1, C1), square(R2, C2),
                                    R2 = R1-1, C2 = C1+1, %diagonally right
                                    occupied(before, R2, C2), %must be occupied
                                    not friendly(white, R2, C2). %must be enemy piece

pawn_moves(white, R1, C1, R2, C2) :- 
                                    square(R1, C1), square(R2, C2),
                                    R2 = R1-1, C2 = C1-1, %diagonal left
                                    occupied(before, R2, C2), %must be occupied
                                    not friendly(white, R2, C2). %must be enemy piece

%black pawn diagonal capture
pawn_moves(black, R1, C1, R2, C2) :- 
                                    square(R1, C1), square(R2, C2),
                                    R2 = R1+1, C2 = C1+1, %diagonal right
                                    occupied(before, R2, C2), %must be occupied
                                    not friendly(black, R2, C2). %must be enemy piece

pawn_moves(black, R1, C1, R2, C2) :- 
                                    square(R1, C1), square(R2, C2),
                                    R2 = R1+1, C2 = C1-1, %diagonal left
                                    occupied(before, R2, C2), %must be occupied
                                    not friendly(black, R2, C2). %must be enemy piece

%rook vertical and horizontal movement
rook_moves(R1, C1, R2, C2):- horizontal(R1, C1, R2, C2).
rook_moves(R1, C1, R2, C2):- vertical(R1, C1, R2, C2).

%bishop diagonal movement
bishop_moves(R1, C1, R2, C2):- diagonal(R1, C1, R2, C2).

%knight movement - L-shape
knight_moves(R1, C1, R2, C2):-
                                square(R1, C1), square(R2, C2),
                                |R1-R2|=1, |C1-C2|=2.
knight_moves(R1, C1, R2, C2):-
                                square(R1, C1), square(R2, C2),
                                |R1-R2|=2, |C1-C2|=1.                             

%---------------------------------------------------------------------------------------------------------------------------
%legal moves - not blocked by another piece, and not occupied by a friendly piece.

%define the pieces that the current piece is between
between_horizontal(R_curr, C_left, C_curr, C_right):-   
                                                    square(R_curr, C_curr), square(R_curr, C_left), square(R_curr, C_right), 
                                                    C_left<C_curr, C_curr<C_right.
between_vertical(C_curr, R_top, R_curr, R_bottom):- square(R_curr, C_curr), square(R_top, C_curr), square(R_bottom, C_curr), 
                                                    R_top<R_curr, R_curr<R_bottom.

between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- 
                                                    square(R_curr, C_curr), square(R1, C1), square(R2, C2), 
                                                    |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, 
                                                    R1<R_curr, R_curr<R2, C1<C_curr, C_curr<C2. %bottom left to top right diagonal
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):-  
                                                    square(R_curr, C_curr), square(R1, C1), square(R2, C2), 
                                                    |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, 
                                                    R2<R_curr, R_curr<R1, C2<C_curr, C_curr<C1. %top right to bottom left diagonal
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- 
                                                    square(R_curr, C_curr), square(R1, C1), square(R2, C2), 
                                                    |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, 
                                                    R1<R_curr, R_curr<R2, C1>C_curr, C_curr>C2. %top left to bottom right diagonal
between_diagonal(R_curr, C_curr, R1, C1, R2, C2):- 
                                                    square(R_curr, C_curr), square(R1, C1), square(R2, C2), 
                                                    |R1-R_curr| = |C1-C_curr|, |R2-R1| = |C2-C1|, 
                                                    R2<R_curr, R_curr<R1, C2>C_curr, C_curr>C1. %bottom right to top left diagonal

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

%after white moves - excluding the black king - if king moves away, it no longer blocks
occupied_excluding_bk(R, C):- 
                            new_placement(ID, R, C), 
                            not placement(black, king, bk, R, C).

%after white moves - excluding the black king - if king moves away, it no longer blocks
%this is for checking the black kings escape squares
blocked_excluding_bk(R1, C1, R2, C2):- 
                            between_horizontal(R1, C1, C_curr, C2), 
                            occupied_excluding_bk(R1, C_curr), R2 = R1.
blocked_excluding_bk(R1, C1, R2, C2):- 
                            between_horizontal(R1, C2, C_curr, C1), 
                            occupied_excluding_bk(R1, C_curr), R2 = R1.
blocked_excluding_bk(R1, C1, R2, C2):- 
                            between_vertical(C1, R1, R_curr, R2), 
                            occupied_excluding_bk(R_curr, C1), C2 = C1.
blocked_excluding_bk(R1, C1, R2, C2):- 
                            between_vertical(C1, R2, R_curr, R1), 
                            occupied_excluding_bk(R_curr, C1), C2 = C1.
blocked_excluding_bk(R1, C1, R2, C2):- 
                            between_diagonal(R_curr, C_curr, R1, C1, R2, C2), 
                            occupied_excluding_bk(R_curr, C_curr).


%helper method 
piece_type(white, ID, T) :- placement(white, T, ID, _, _).


%squares the black king threatens
black_threatens(R2, C2):- 
                        placement(black, king, bk, R1, C1), %where black king is originally
                        king_moves(R1, C1, R2, C2).

%squares the white king threatens
white_threatens(R2, C2):- 
                        new_placement(wk, R1, C1),
                        king_moves(R1, C1, R2, C2).
white_threatens(R2, C2):- 
                        new_placement(ID, R1, C1),
                        piece_type(white, ID, queen),
                        queen_moves(R1, C1, R2, C2),
                        not blocked_excluding_bk(R1, C1, R2, C2),
                        (R1, C1) != (R2, C2),
                        not new_placement(ID, R2, C2).

white_threatens(R2, C2):- 
                        new_placement(ID, R1, C1),  %where white king is after move
                        piece_type(white, ID, pawn),
                        R2 = R1-1, C2 = C1+1,
                        square(R2, C2). %pawns threaten diagonally forward
white_threatens(R2, C2):- 
                        new_placement(ID, R1, C1),  %where white king is after move
                        piece_type(white, ID, pawn),
                        R2 = R1-1, C2 = C1-1,
                        square(R2, C2). %pawns threaten diagonally forward


white_threatens(R2, C2):- 
                        new_placement(ID, R1, C1),
                        piece_type(white, ID, rook),
                        rook_moves(R1, C1, R2, C2),
                        not blocked_excluding_bk(R1, C1, R2, C2).
white_threatens(R2, C2):- 
                        new_placement(ID, R1, C1),
                        piece_type(white, ID, bishop),
                        bishop_moves(R1, C1, R2, C2),
                        not blocked_excluding_bk(R1, C1, R2, C2).
white_threatens(R2, C2):-
                        new_placement(ID, R1, C1),
                        piece_type(white, ID, knight),
                        knight_moves(R1, C1, R2, C2),
                        not blocked_excluding_bk(R1, C1, R2, C2).

%defining legal moves - not blocked. only modelling for white pieces as we want to find a winning move for white, so we assume black will play optimally and move their king out of check if possible.
legal_queen_move(P, S, R1, C1, R2, C2):- 
                                state(S),
                                player(P),
                                queen_moves(R1, C1, R2, C2), %valid move for a queen
                                not blocked(S, R1, C1, R2, C2), % not blocked by another piece
                                not friendly(P, R2, C2). % not occupied by a friendly piece

legal_king_move(P, R1, C1, R2, C2):- 
                                player(P),  
                                king_moves(R1, C1, R2, C2), % valid move for a king
                                not friendly(P, R2, C2).% not occupied by another piece 

legal_pawn_move(P, S, R1, C1, R2, C2):- 
                                player(P), 
                                state(S),
                                pawn_moves(P, R1, C1, R2, C2), %valid move for a pawn
                                not friendly(P, R2, C2). % not occupied by a friendly piece
legal_rook_move(P, S, R1, C1, R2, C2):-
                                player(P), 
                                state(S),
                                rook_moves(R1, C1, R2, C2),
                                not blocked(S, R1, C1, R2, C2),
                                not friendly(P, R2, C2).
legal_bishop_move(P, S, R1, C1, R2, C2):-
                                player(P), 
                                state(S),
                                bishop_moves(R1, C1, R2, C2),
                                not blocked(S, R1, C1, R2, C2),
                                not friendly(P, R2, C2).
legal_knight_move(P, S, R1, C1, R2, C2):-
                                player(P), 
                                state(S),
                                knight_moves(R1, C1, R2, C2),
                                not blocked(S, R1, C1, R2, C2),
                                not friendly(P, R2, C2).

%piece must be placed on the board to move
legal_move(ID, R1, C1, R2, C2):- 
                                placement(white, queen, ID, R1, C1), 
                                legal_queen_move(white, before, R1, C1, R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
                                placement(white, king, ID, R1, C1), 
                                legal_king_move(white, R1, C1, R2, C2), 
                                not black_threatens(R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
                                placement(black, king, ID, R1, C1), 
                                legal_king_move(black, R1, C1, R2, C2), 
                                not white_threatens(R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
                                placement(white, pawn, ID, R1, C1),
                                legal_pawn_move(white, before, R1, C1, R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
                                placement(white, rook, ID, R1, C1),
                                legal_rook_move(white, before, R1, C1, R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
                                placement(white, bishop, ID, R1, C1),
                                legal_bishop_move(white, before, R1, C1, R2, C2).
legal_move(ID, R1, C1, R2, C2):- 
                                placement(white, knight, ID, R1, C1),
                                legal_knight_move(white, before, R1, C1, R2, C2).

%=============================================================
% MATE-IN-ONE SOLVER EXTENSION
%=============================================================

piece_type(black, ID, T) :- placement(black, T, ID, _, _).

% Choose exactly one legal white move
{ chosen_move(ID, R1, C1, R2, C2) : legal_move(ID, R1, C1, R2, C2) } = 1.

moved_piece(ID) :- chosen_move(ID, _, _, _, _).

% Compute new placements after white's move
new_placement(ID, R2, C2) :- chosen_move(ID, _, _, R2, C2).

new_placement(ID, R, C) :- 
    placement(white, _, ID, R, C), 
    not moved_piece(ID).

% Track captured black pieces
captured_at(R2, C2) :- chosen_move(_, _, _, R2, C2).

new_placement(ID, R, C) :- 
    placement(black, _, ID, R, C), 
    not captured_at(R, C).

% Black king position (unchanged by white's move)
bk_pos(R, C) :- placement(black, king, bk, R, C).

% White must give check
in_check :- bk_pos(R, C), white_threatens(R, C).
:- not in_check.

% Black non-king pieces still on board
black_piece_pos(ID, R, C) :- 
    placement(black, _, ID, R, C), 
    ID != bk,
    not captured_at(R, C).

black_nonking_at(R, C) :- black_piece_pos(_, R, C).

white_piece_at(R, C) :- new_placement(ID, R, C), placement(white, _, ID, _, _).

% Candidate escape squares for black king
candidate_escape(R2, C2) :-
    bk_pos(R1, C1),
    king_moves(R1, C1, R2, C2),
    square(R2, C2).

% King can escape to empty square not threatened by white
king_escape_empty(R2, C2) :-
    candidate_escape(R2, C2),
    not black_nonking_at(R2, C2),
    not white_piece_at(R2, C2),
    not white_threatens(R2, C2).

%--------------------------------------------------------------
% Check if king can capture a white piece at (ER,EC)
% We need white_threatens_excl(R2,C2,ER,EC) = white threatens (R2,C2)
% when the white piece at (ER,EC) is removed (king captures it)
% All variables must be grounded via square/2
%--------------------------------------------------------------

% Board occupancy after white move, excluding square (ER,EC)
occupied_after_excl(R, C, ER, EC) :-
    new_placement(_, R, C),
    square(ER, EC),
    (R, C) != (ER, EC).

blocked_after_excl(R1, C1, R2, C2, ER, EC) :-
    between_horizontal(R1, C1, C_curr, C2),
    occupied_after_excl(R1, C_curr, ER, EC),
    R2 = R1,
    square(ER, EC).
blocked_after_excl(R1, C1, R2, C2, ER, EC) :-
    between_horizontal(R1, C2, C_curr, C1),
    occupied_after_excl(R1, C_curr, ER, EC),
    R2 = R1,
    square(ER, EC).
blocked_after_excl(R1, C1, R2, C2, ER, EC) :-
    between_vertical(C1, R1, R_curr, R2),
    occupied_after_excl(R_curr, C1, ER, EC),
    C2 = C1,
    square(ER, EC).
blocked_after_excl(R1, C1, R2, C2, ER, EC) :-
    between_vertical(C1, R2, R_curr, R1),
    occupied_after_excl(R_curr, C1, ER, EC),
    C2 = C1,
    square(ER, EC).
blocked_after_excl(R1, C1, R2, C2, ER, EC) :-
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2),
    occupied_after_excl(R_curr, C_curr, ER, EC),
    square(ER, EC).

% white_threatens_excl(R2,C2,ER,EC): white threatens (R2,C2) even after
% removing the white piece at (ER,EC) (i.e., when black king captures it)
% ER,EC grounded via white_piece_at
white_threatens_excl(R2, C2, ER, EC) :-
    white_piece_at(ER, EC),
    new_placement(wk, R1, C1),
    (R1, C1) != (ER, EC),
    king_moves(R1, C1, R2, C2).

white_threatens_excl(R2, C2, ER, EC) :-
    white_piece_at(ER, EC),
    new_placement(ID, R1, C1),
    piece_type(white, ID, rook),
    (R1, C1) != (ER, EC),
    rook_moves(R1, C1, R2, C2),
    not blocked_after_excl(R1, C1, R2, C2, ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    white_piece_at(ER, EC),
    new_placement(ID, R1, C1),
    piece_type(white, ID, bishop),
    (R1, C1) != (ER, EC),
    bishop_moves(R1, C1, R2, C2),
    not blocked_after_excl(R1, C1, R2, C2, ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    white_piece_at(ER, EC),
    new_placement(ID, R1, C1),
    piece_type(white, ID, queen),
    (R1, C1) != (ER, EC),
    queen_moves(R1, C1, R2, C2),
    not blocked_after_excl(R1, C1, R2, C2, ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    white_piece_at(ER, EC),
    new_placement(ID, R1, C1),
    piece_type(white, ID, knight),
    (R1, C1) != (ER, EC),
    knight_moves(R1, C1, R2, C2).

white_threatens_excl(R2, C2, ER, EC) :-
    white_piece_at(ER, EC),
    new_placement(ID, R1, C1),
    piece_type(white, ID, pawn),
    (R1, C1) != (ER, EC),
    R2 = R1-1, C2 = C1+1,
    square(R2, C2).

white_threatens_excl(R2, C2, ER, EC) :-
    white_piece_at(ER, EC),
    new_placement(ID, R1, C1),
    piece_type(white, ID, pawn),
    (R1, C1) != (ER, EC),
    R2 = R1-1, C2 = C1-1,
    square(R2, C2).

% King can escape by capturing a white piece if the destination is not still threatened
king_escape_capture(R2, C2) :-
    candidate_escape(R2, C2),
    white_piece_at(R2, C2),
    not black_nonking_at(R2, C2),
    not white_threatens_excl(R2, C2, R2, C2).

% King can escape by capturing a black piece (own piece - illegal) - skip
% King can escape by moving to square with black piece (can't - friendly)
% King can escape to square occupied by black non-king piece by capturing it
king_escape_capture_black(R2, C2) :-
    candidate_escape(R2, C2),
    black_nonking_at(R2, C2),
    not white_piece_at(R2, C2),
    not white_threatens(R2, C2).

king_can_escape :- king_escape_empty(_, _).
king_can_escape :- king_escape_capture(_, _).
king_can_escape :- king_escape_capture_black(_, _).

:- king_can_escape.

% Also check: can black block or capture the checking piece?
% Identify checking pieces
checking_piece(ID, CR, CC) :-
    new_placement(ID, CR, CC),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, rook),
    rook_moves(CR, CC, BKR, BKC),
    not blocked_excluding_bk(CR, CC, BKR, BKC).

checking_piece(ID, CR, CC) :-
    new_placement(ID, CR, CC),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, queen),
    queen_moves(CR, CC, BKR, BKC),
    not blocked_excluding_bk(CR, CC, BKR, BKC).

checking_piece(ID, CR, CC) :-
    new_placement(ID, CR, CC),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, bishop),
    bishop_moves(CR, CC, BKR, BKC),
    not blocked_excluding_bk(CR, CC, BKR, BKC).

checking_piece(ID, CR, CC) :-
    new_placement(ID, CR, CC),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, knight),
    knight_moves(CR, CC, BKR, BKC).

checking_piece(ID, CR, CC) :-
    new_placement(ID, CR, CC),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, pawn),
    BKR = CR-1, BKC = CC+1.

checking_piece(ID, CR, CC) :-
    new_placement(ID, CR, CC),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, pawn),
    BKR = CR-1, BKC = CC-1.

double_check :- 
    checking_piece(ID1, R1, C1), 
    checking_piece(ID2, R2, C2), 
    (ID1, R1, C1) != (ID2, R2, C2).

% Interposition squares (only relevant for single check by sliding piece)
interpose_sq(R, C) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    not double_check,
    between_horizontal(CR, CC, C, BKC),
    R = CR,
    square(R, C).

interpose_sq(R, C) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    not double_check,
    between_horizontal(CR, BKC, C, CC),
    R = CR,
    square(R, C).

interpose_sq(R, C) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    not double_check,
    between_vertical(CC, CR, R, BKR),
    C = CC,
    square(R, C).

interpose_sq(R, C) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    not double_check,
    between_vertical(CC, BKR, R, CR),
    C = CC,
    square(R, C).

interpose_sq(R, C) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    not double_check,
    between_diagonal(R, C, CR, CC, BKR, BKC),
    square(R, C).

% Can black capture the checking piece?
black_can_capture_checker :-
    checking_piece(_, CR, CC),
    not double_check,
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, rook),
    rook_moves(BR, BC, CR, CC),
    not blocked(before, BR, BC, CR, CC).

black_can_capture_checker :-
    checking_piece(_, CR, CC),
    not double_check,
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, bishop),
    bishop_moves(BR, BC, CR, CC),
    not blocked(before, BR, BC, CR, CC).

black_can_capture_checker :-
    checking_piece(_, CR, CC),
    not double_check,
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, queen),
    queen_moves(BR, BC, CR, CC),
    not blocked(before, BR, BC, CR, CC).

black_can_capture_checker :-
    checking_piece(_, CR, CC),
    not double_check,
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, knight),
    knight_moves(BR, BC, CR, CC).

black_can_capture_checker :-
    checking_piece(_, CR, CC),
    not double_check,
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, pawn),
    pawn_moves(black, BR, BC, CR, CC).

% Can black interpose?
black_can_interpose :-
    interpose_sq(IR, IC),
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, rook),
    rook_moves(BR, BC, IR, IC),
    not blocked(before, BR, BC, IR, IC).

black_can_interpose :-
    interpose_sq(IR, IC),
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, bishop),
    bishop_moves(BR, BC, IR, IC),
    not blocked(before, BR, BC, IR, IC).

black_can_interpose :-
    interpose_sq(IR, IC),
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, queen),
    queen_moves(BR, BC, IR, IC),
    not blocked(before, BR, BC, IR, IC).

black_can_interpose :-
    interpose_sq(IR, IC),
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, knight),
    knight_moves(BR, BC, IR, IC).

black_can_interpose :-
    interpose_sq(IR, IC),
    black_piece_pos(BID, BR, BC),
    piece_type(black, BID, pawn),
    pawn_moves(black, BR, BC, IR, IC).

:- black_can_capture_checker.
:- black_can_interpose.

% Output the mating move
mate_move(Color, Type, R1, C1, R2, C2) :-
    to_move(Color),
    chosen_move(ID, R1, C1, R2, C2),
    placement(Color, Type, ID, R1, C1).

#show mate_move/6.