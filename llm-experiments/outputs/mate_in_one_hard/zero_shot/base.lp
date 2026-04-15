%define the chess board
rows(1..8).
cols(1..8).
square(R, C):- rows(R), cols(C).

%define players
player(white; black).

%define chess pieces
pieceType(king; queen; pawn; rook; bishop; knight).

%define the initial positions of the pieces on the board
placement(white, king, wk, 8, 2).
placement(white, rook, wr1, 8, 4).
placement(white, rook, wr2, 7, 7).
placement(white, queen, wq, 2, 6).
placement(white, pawn, wp1, 5, 1).
placement(white, pawn, wp2, 7, 2).
placement(white, pawn, wp3, 7, 3).
placement(white, pawn, wp4, 6, 3).
placement(white, pawn, wp5, 7, 5).
placement(white, pawn, wp6, 7, 6).
placement(white, pawn, wp7, 4, 6).
placement(white, knight, wn1, 6, 2).
placement(white, knight, wn2, 6, 5).
placement(white, bishop, wb1, 6, 6).

placement(black, rook, br1, 1, 1).
placement(black, rook, br2, 1, 8).
placement(black, bishop, bb1, 1, 3).
placement(black, bishop, bb2, 1, 6).
placement(black, pawn, bp1, 2, 1).
placement(black, pawn, bp2, 2, 2).
placement(black, pawn, bp3, 5, 3).
placement(black, pawn, bp4, 4, 4).
placement(black, pawn, bp5, 4, 5).
placement(black, pawn, bp6, 2, 7).
placement(black, pawn, bp7, 3, 8).
placement(black, pawn, bp8, 4, 8).
placement(black, knight, bn1, 3, 3).
placement(black, knight, bn2, 3, 5).
placement(black, king, bk, 3, 4).
placement(black, queen, bb1, 8, 8).


%piece next to move
to_move(white).

%define the legal moves
horizontal(R, C1, R, C2):- 
                    square(R, C1), 
                    square(R, C2), 
                    C1!=C2. %horizontal movement
vertical(R1, C, R2, C):-  
                    square(R1, C), 
                    square(R2, C), 
                    R1!=R2. %vertical movement
diagonal(R1, C1, R2, C2):- 
                    square(R1, C1), 
                    square(R2, C2), 
                    |R1-R2| = |C1-C2|, 
                    (R1, C1) != (R2, C2). %diagonal movement - absolute differences are equal, pairs arent equal

%defining legal moves
%Queen moves - hor, ver, diag - any no. of squares
queen_moves(R1, C1, R2, C2):- horizontal(R1, C1, R2, C2).
queen_moves(R1, C1, R2, C2):- vertical(R1, C1, R2, C2).
queen_moves(R1, C1, R2, C2):- diagonal(R1, C1, R2, C2).

%King moves - to adjacent squares in any direction
king_moves(R1, C1, R2, C2):- 
                    square(R1, C1),  
                    square(R2, C2), 
                    |C1-C2| <= 1, 
                    |R1-R2| <=1, 
                    (R1, C1) != (R2, C2).

%pawn moves - only forward, and can only capture diagonally
pawn_moves(white, R1, C1, R2, C2):- 
                    square(R1, C1), square(R2, C2), 
                    R2 = R1-1, C2 = C1, %forward move
                    not occupied(before, R2, C2). %cannot move forward if occupied

pawn_moves(black, R1, C1, R2, C2):-
                    square(R1, C1), square(R2, C2), 
                    R2 = R1+1, C2=C1, %forward move
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
between_vertical(C_curr, R_top, R_curr, R_bottom):- 
                    square(R_curr, C_curr), square(R_top, C_curr), square(R_bottom, C_curr), 
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

%friendly pieces after the move - to stop a player from taking their own pieces
friendly_after(P, R, C):- new_placement(ID, R, C), placement(P, _, ID, _, _).


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

%squared the white queen threatens
white_threatens(R2, C2):- 
                    new_placement(ID, R1, C1),
                    piece_type(white, ID, queen),
                    queen_moves(R1, C1, R2, C2),
                    not blocked_excluding_bk(R1, C1, R2, C2),
                    (R1, C1) != (R2, C2),
                    not new_placement(ID, R2, C2).
%squared the white pawns threaten
white_threatens(R2, C2):- 
                    new_placement(ID, R1, C1),  %where white pawn is after move
                    piece_type(white, ID, pawn),
                    R2 = R1-1, C2 = C1+1,
                    square(R2, C2). %pawns threaten diagonally forward
white_threatens(R2, C2):- 
                    new_placement(ID, R1, C1),
                    piece_type(white, ID, pawn),
                    R2 = R1-1, C2 = C1-1,
                    square(R2, C2). %pawns threaten diagonally forward

%squared the white rook threatens
white_threatens(R2, C2):- 
                    new_placement(ID, R1, C1),
                    piece_type(white, ID, rook),
                    rook_moves(R1, C1, R2, C2),
                    not blocked_excluding_bk(R1, C1, R2, C2).

%squared the white bishops threaten
white_threatens(R2, C2):- 
                    new_placement(ID, R1, C1),
                    piece_type(white, ID, bishop),
                    bishop_moves(R1, C1, R2, C2),
                    not blocked_excluding_bk(R1, C1, R2, C2).

%squares the white knight threaten
white_threatens(R2, C2):-
                    new_placement(ID, R1, C1),
                    piece_type(white, ID, knight),
                    knight_moves(R1, C1, R2, C2).

%defining legal moves - not blocked. only modelling for white pieces as we want to find a winning move for white, so we assume black will play optimally and move their king out of check if possible.
legal_queen_move(P, S, R1, C1, R2, C2):- 
                    state(S),
                    player(P),
                    queen_moves(R1, C1, R2, C2), %valid move for a queen
                    not blocked(S, R1, C1, R2, C2), % not blocked by another piece
                    not friendly(P, R2, C2). % not occupied by a friendly piece

legal_king_move(white, R1, C1, R2, C2):- 
                    king_moves(R1, C1, R2, C2), % valid move for a king
                    not friendly(white, R2, C2).% not occupied by another piece - before state for move generation

legal_king_move(black, R1, C1, R2, C2):- 
                    king_moves(R1, C1, R2, C2), % valid move for a king
                    not friendly_after(black, R2, C2).% after state for escape checking

legal_pawn_move(P, S, R1, C1, R2, C2):- 
                    player(P), 
                    state(S),
                    pawn_moves(P, R1, C1, R2, C2), %valid move for a pawn
                    not friendly(P, R2, C2). % not occupied by a friendly piece
legal_rook_move(P, S, R1, C1, R2, C2):-
                    player(P), 
                    state(S),
                    rook_moves(R1, C1, R2, C2), %valid move for a rook
                    not blocked(S, R1, C1, R2, C2),
                    not friendly(P, R2, C2).
legal_bishop_move(P, S, R1, C1, R2, C2):-
                    player(P), 
                    state(S),
                    bishop_moves(R1, C1, R2, C2), %valid move for a bishop
                    not blocked(S, R1, C1, R2, C2),
                    not friendly(P, R2, C2).
legal_knight_move(P, S, R1, C1, R2, C2):-
                    player(P), 
                    state(S),
                    knight_moves(R1, C1, R2, C2), %valid move for a knight
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
