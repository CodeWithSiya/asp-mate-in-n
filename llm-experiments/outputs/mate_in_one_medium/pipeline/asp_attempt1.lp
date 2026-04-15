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

% Choose exactly one legal white move to make
{ chosen_move(ID, R1, C1, R2, C2) : legal_move(ID, R1, C1, R2, C2) } = 1.

% Compute new placements after the chosen white move
% The moved piece goes to the new square
new_placement(ID, R2, C2) :- chosen_move(ID, R1, C1, R2, C2).
% All other white pieces stay where they are
new_placement(ID, R, C) :- placement(white, _, ID, R, C), not chosen_move(ID, R, C, _, _), not chosen_move(ID, _, _, _, _).
% White pieces that didn't move stay
new_placement(ID, R, C) :- placement(white, _, ID, R, C), chosen_move(OID, _, _, _, _), OID != ID.

% Black pieces remain unless captured
new_placement(ID, R, C) :- placement(black, _, ID, R, C), not chosen_move(_, _, _, R, C).

% piece_type for all pieces (including black, for completeness)
piece_type(black, ID, T) :- placement(black, T, ID, _, _).

% After white's move, determine if black king is in check
% Black king position (it doesn't move in white's turn)
bk_pos(R, C) :- placement(black, king, bk, R, C).

% White threatens the black king's square => black king is in check
in_check :- bk_pos(R, C), white_threatens(R, C).

% Black king must be in check for it to be checkmate
:- not in_check.

% Now determine if black king has any legal escape
% Black king can move to adjacent squares not threatened by white and not occupied by black pieces
% (it can capture white pieces if that square is not threatened)

% Squares occupied by black pieces (after white's move, excluding the black king itself)
black_occupied(R, C) :- placement(black, _, ID, R, C), ID != bk.

% A square is a candidate escape if:
% 1. It's adjacent to the black king
% 2. It's on the board
% 3. It's not occupied by a black piece (king can't capture own pieces)
candidate_escape(R2, C2) :-
    bk_pos(R1, C1),
    king_moves(R1, C1, R2, C2),
    square(R2, C2),
    not black_occupied(R2, C2).

% An escape square is safe if white does not threaten it after white's move
% (white_threatens already accounts for the new placement)
% But we need to handle the case where the black king captures a white piece:
% if king captures a white piece, that white piece is removed, potentially unblocking attacks

% For simplicity and correctness, we check if white_threatens the escape square
% white_threatens uses new_placement which includes white pieces after the move
% If black king captures a white piece at (R2,C2), that piece is no longer there
% We need to recompute threats excluding the captured piece

% White piece at escape square (could be captured by black king)
white_piece_at(R, C) :- new_placement(ID, R, C), placement(white, _, ID, _, _).

% For each candidate escape, check if it's safe
% We need to determine if white threatens (R2,C2) even after removing any white piece there
% (since the black king would capture it)

% Recompute white threats excluding a captured white piece at (ER, EC)
% We'll use a separate predicate for this

% new_placement_excl(ID, R, C, ER, EC): new placement excluding white piece at (ER,EC)
new_placement_excl(ID, R, C, ER, EC) :-
    new_placement(ID, R, C),
    not (R = ER, C = EC).

% blocked_excl: blocking excluding the captured piece
occupied_excl(R, C, ER, EC) :- new_placement_excl(_, R, C, ER, EC).

blocked_excl(R1, C1, R2, C2, ER, EC) :-
    between_horizontal(R1, C1, C_curr, C2),
    occupied_excl(R1, C_curr, ER, EC), R2 = R1.
blocked_excl(R1, C1, R2, C2, ER, EC) :-
    between_horizontal(R1, C2, C_curr, C1),
    occupied_excl(R1, C_curr, ER, EC), R2 = R1.
blocked_excl(R1, C1, R2, C2, ER, EC) :-
    between_vertical(C1, R1, R_curr, R2),
    occupied_excl(R_curr, C1, ER, EC), C2 = C1.
blocked_excl(R1, C1, R2, C2, ER, EC) :-
    between_vertical(C1, R2, R_curr, R1),
    occupied_excl(R_curr, C1, ER, EC), C2 = C1.
blocked_excl(R1, C1, R2, C2, ER, EC) :-
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2),
    occupied_excl(R_curr, C_curr, ER, EC).

% White threatens (R2,C2) after removing piece at (ER,EC)
white_threatens_excl(R2, C2, ER, EC) :-
    new_placement_excl(wk, R1, C1, ER, EC),
    king_moves(R1, C1, R2, C2).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement_excl(ID, R1, C1, ER, EC),
    piece_type(white, ID, rook),
    rook_moves(R1, C1, R2, C2),
    not blocked_excl(R1, C1, R2, C2, ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement_excl(ID, R1, C1, ER, EC),
    piece_type(white, ID, bishop),
    bishop_moves(R1, C1, R2, C2),
    not blocked_excl(R1, C1, R2, C2, ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement_excl(ID, R1, C1, ER, EC),
    piece_type(white, ID, queen),
    queen_moves(R1, C1, R2, C2),
    not blocked_excl(R1, C1, R2, C2, ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement_excl(ID, R1, C1, ER, EC),
    piece_type(white, ID, knight),
    knight_moves(R1, C1, R2, C2),
    not blocked_excl(R1, C1, R2, C2, ER, EC).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement_excl(ID, R1, C1, ER, EC),
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1+1,
    square(R2, C2).

white_threatens_excl(R2, C2, ER, EC) :-
    new_placement_excl(ID, R1, C1, ER, EC),
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1-1,
    square(R2, C2).

% An escape is valid if the square is not threatened by white
% considering whether a white piece is captured there
escape_possible(R2, C2) :-
    candidate_escape(R2, C2),
    not white_piece_at(R2, C2),
    not white_threatens(R2, C2).

escape_possible(R2, C2) :-
    candidate_escape(R2, C2),
    white_piece_at(R2, C2),
    not white_threatens_excl(R2, C2, R2, C2).

% For checkmate, no escape must be possible
:- escape_possible(_, _).

% Also check that no black piece can block the check or capture the checking piece
% (For a complete mate-in-one solver, we should verify no black move resolves check)
% We handle king moves above; for other black pieces blocking/capturing:

% Find the checking white piece(s)
checking_piece(ID, R1, C1) :-
    new_placement(ID, R1, C1),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    white_threatens(BKR, BKC),
    % This piece attacks the black king
    piece_type(white, ID, rook),
    rook_moves(R1, C1, BKR, BKC),
    not blocked_excluding_bk(R1, C1, BKR, BKC).

checking_piece(ID, R1, C1) :-
    new_placement(ID, R1, C1),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, queen),
    queen_moves(R1, C1, BKR, BKC),
    not blocked_excluding_bk(R1, C1, BKR, BKC).

checking_piece(ID, R1, C1) :-
    new_placement(ID, R1, C1),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, bishop),
    bishop_moves(R1, C1, BKR, BKC),
    not blocked_excluding_bk(R1, C1, BKR, BKC).

checking_piece(ID, R1, C1) :-
    new_placement(ID, R1, C1),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, knight),
    knight_moves(R1, C1, BKR, BKC).

checking_piece(ID, R1, C1) :-
    new_placement(ID, R1, C1),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, pawn),
    BKR = R1-1, BKC = C1+1.

checking_piece(ID, R1, C1) :-
    new_placement(ID, R1, C1),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    piece_type(white, ID, pawn),
    BKR = R1-1, BKC = C1-1.

% Double check: if two pieces give check, only king moves can resolve it
double_check :- checking_piece(ID1, _, _), checking_piece(ID2, _, _), ID1 != ID2.

% Squares between checking piece and black king (for blocking)
interpose_square(R, C) :-
    checking_piece(CID, CR, CC),
    bk_pos(BKR, BKC),
    between_horizontal(CR, CC, C, BKC), R = CR, not double_check.
interpose_square(R, C) :-
    checking_piece(CID, CR, CC),
    bk_pos(BKR, BKC),
    between_horizontal(CR, BKC, C, CC), R = CR, not double_check.
interpose_square(R, C) :-
    checking_piece(CID, CR, CC),
    bk_pos(BKR, BKC),
    between_vertical(CC, CR, R, BKR), C = CC, not double_check.
interpose_square(R, C) :-
    checking_piece(CID, CR, CC),
    bk_pos(BKR, BKC),
    between_vertical(CC, BKR, R, CR), C = CC, not double_check.
interpose_square(R, C) :-
    checking_piece(CID, CR, CC),
    bk_pos(BKR, BKC),
    between_diagonal(R, C, CR, CC, BKR, BKC), not double_check.

% Can a black piece (non-king) capture the checking piece?
black_can_capture_checker :-
    checking_piece(CID, CR, CC),
    not double_check,
    placement(black, rook, BID, BR, BC), BID != bk,
    rook_moves(BR, BC, CR, CC),
    not blocked(before, BR, BC, CR, CC).

black_can_capture_checker :-
    checking_piece(CID, CR, CC),
    not double_check,
    placement(black, bishop, BID, BR, BC), BID != bk,
    bishop_moves(BR, BC, CR, CC),
    not blocked(before, BR, BC, CR, CC).

black_can_capture_checker :-
    checking_piece(CID, CR, CC),
    not double_check,
    placement(black, queen, BID, BR, BC), BID != bk,
    queen_moves(BR, BC, CR, CC),
    not blocked(before, BR, BC, CR, CC).

black_can_capture_checker :-
    checking_piece(CID, CR, CC),
    not double_check,
    placement(black, knight, BID, BR, BC), BID != bk,
    knight_moves(BR, BC, CR, CC).

black_can_capture_checker :-
    checking_piece(CID, CR, CC),
    not double_check,
    placement(black, pawn, BID, BR, BC), BID != bk,
    pawn_moves(black, BR, BC, CR, CC).

% Can a black piece (non-king) interpose?
black_can_interpose :-
    interpose_square(IR, IC),
    placement(black, rook, BID, BR, BC), BID != bk,
    rook_moves(BR, BC, IR, IC),
    not blocked(before, BR, BC, IR, IC).

black_can_interpose :-
    interpose_square(IR, IC),
    placement(black, bishop, BID, BR, BC), BID != bk,
    bishop_moves(BR, BC, IR, IC),
    not blocked(before, BR, BC, IR, IC).

black_can_interpose :-
    interpose_square(IR, IC),
    placement(black, queen, BID, BR, BC), BID != bk,
    queen_moves(BR, BC, IR, IC),
    not blocked(before, BR, BC, IR, IC).

black_can_interpose :-
    interpose_square(IR, IC),
    placement(black, knight, BID, BR, BC), BID != bk,
    knight_moves(BR, BC, IR, IC).

black_can_interpose :-
    interpose_square(IR, IC),
    placement(black, pawn, BID, BR, BC), BID != bk,
    pawn_moves(black, BR, BC, IR, IC).

% If black can capture the checker or interpose, it's not checkmate
:- black_can_capture_checker.
:- black_can_interpose.

% Output the mating move
mate_move(Color, Type, R1, C1, R2, C2) :-
    to_move(Color),
    chosen_move(ID, R1, C1, R2, C2),
    placement(Color, Type, ID, R1, C1).

#show mate_move/6.