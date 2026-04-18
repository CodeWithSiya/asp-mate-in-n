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

%==========================================================
% EXTENSION: Mate-in-one reasoning
%==========================================================

% 1. Choose exactly one legal white move
1 { choose_move(ID, R1, C1, R2, C2) : legal_move(ID, R1, C1, R2, C2) } 1.

% 2. Derive new_placement after the chosen move
% The moved piece goes to its new square
new_placement(ID, R2, C2) :- choose_move(ID, R1, C1, R2, C2).

% All other white pieces stay where they are
new_placement(ID, R, C) :- 
    placement(white, _, ID, R, C),
    not choose_move(ID, _, _, _, _).

% All black pieces stay where they are, unless captured
new_placement(ID, R, C) :- 
    placement(black, _, ID, R, C),
    not captured_piece(ID).

% A black piece is captured if the chosen move lands on its square
captured_piece(ID) :- 
    choose_move(_, _, _, R2, C2),
    placement(black, _, ID, R2, C2).

% 3. Check detection: black king is in check after white's move
black_king_in_check :- 
    placement(black, king, bk, R, C),
    white_threatens(R, C).

% 4. Determine the black king's position after white moves
bk_pos(R, C) :- placement(black, king, bk, R, C).

% 5. Squares the black king can potentially move to (legal king moves after white's move)
% The black king can move to a square if:
%   - it's a valid king move
%   - not occupied by a friendly (black) piece after white's move
%   - not threatened by white after white's move (and after king moves there)

% Black king escape: king moves to (R2,C2)
% We need to check if (R2,C2) is safe for the black king
% i.e., not attacked by white when king is at (R2,C2)

% First, check if the destination is occupied by a black piece (after white's move)
black_piece_at_after(R, C) :- 
    new_placement(ID, R, C),
    placement(black, _, ID, _, _).

% The black king can attempt to move to (R2,C2)
bk_can_move_to(R2, C2) :- 
    bk_pos(R1, C1),
    legal_king_move(black, R1, C1, R2, C2),
    not white_threatens(R2, C2),
    not white_king_threatens_after(R2, C2).

% White king threatens squares after the move (already covered by white_threatens)
% but we need to be careful: white_threatens uses new_placement which already has the moved piece
% white_threatens covers the white king via new_placement(wk,...)

% Actually white_threatens already handles all white pieces including king
% So bk_can_move_to is correct as written above

% 6. Check if black has any escape by moving the king
bk_has_king_escape :- bk_can_move_to(_, _).

% 7. Check if black can block or capture the checking piece
% For this we need to identify what's giving check and whether black pieces
% (other than the king) can interpose or capture

% Identify the checking piece(s)
checking_piece(ID, R1, C1) :- 
    new_placement(ID, R1, C1),
    placement(white, _, ID, _, _),
    bk_pos(BKR, BKC),
    white_piece_attacks_sq(ID, R1, C1, BKR, BKC).

% White piece attacks a square (used for check detection)
white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    piece_type(white, ID, queen),
    queen_moves(R1, C1, R2, C2),
    not blocked_excluding_bk(R1, C1, R2, C2).

white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    piece_type(white, ID, rook),
    rook_moves(R1, C1, R2, C2),
    not blocked_excluding_bk(R1, C1, R2, C2).

white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    piece_type(white, ID, bishop),
    bishop_moves(R1, C1, R2, C2),
    not blocked_excluding_bk(R1, C1, R2, C2).

white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    piece_type(white, ID, knight),
    knight_moves(R1, C1, R2, C2).

white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1+1, square(R2, C2).

white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1-1, square(R2, C2).

white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    piece_type(white, ID, king),
    king_moves(R1, C1, R2, C2).

% Squares between the checking piece and the black king (for interposition)
% Only relevant for sliding pieces (queen, rook, bishop)

% Interposition squares: squares between checker and black king
interpose_sq(R, C) :-
    checking_piece(ID, CR, CC),
    bk_pos(BKR, BKC),
    piece_type(white, ID, queen),
    between_on_line(CR, CC, BKR, BKC, R, C).

interpose_sq(R, C) :-
    checking_piece(ID, CR, CC),
    bk_pos(BKR, BKC),
    piece_type(white, ID, rook),
    between_on_line(CR, CC, BKR, BKC, R, C).

interpose_sq(R, C) :-
    checking_piece(ID, CR, CC),
    bk_pos(BKR, BKC),
    piece_type(white, ID, bishop),
    between_on_line(CR, CC, BKR, BKC, R, C).

% Squares strictly between two squares on the same line
between_on_line(R1, C1, R2, C2, R, C) :-
    square(R, C),
    % horizontal
    R1 = R2, R = R1,
    C1 < C2, C1 < C, C < C2.

between_on_line(R1, C1, R2, C2, R, C) :-
    square(R, C),
    % horizontal other direction
    R1 = R2, R = R1,
    C2 < C1, C2 < C, C < C1.

between_on_line(R1, C1, R2, C2, R, C) :-
    square(R, C),
    % vertical
    C1 = C2, C = C1,
    R1 < R2, R1 < R, R < R2.

between_on_line(R1, C1, R2, C2, R, C) :-
    square(R, C),
    % vertical other direction
    C1 = C2, C = C1,
    R2 < R1, R2 < R, R < R1.

between_on_line(R1, C1, R2, C2, R, C) :-
    square(R, C),
    % diagonal
    |R1-R2| = |C1-C2|,
    |R1-R| = |C1-C|,
    |R2-R| = |C2-C|,
    R != R1, R != R2,
    % R is strictly between R1 and R2
    ((R1 < R, R < R2) ; (R2 < R, R < R1)).

% Black pieces (non-king) that can capture the checking piece
black_can_capture_checker :-
    checking_piece(ID, CR, CC),
    black_piece_can_reach(BID, CR, CC),
    BID != bk,
    % capturing doesn't leave black king in check from another white piece
    not capture_leaves_bk_in_check(BID, CR, CC).

% Black pieces (non-king) that can interpose
black_can_interpose :-
    interpose_sq(R, C),
    black_piece_can_reach(BID, R, C),
    BID != bk,
    not interpose_leaves_bk_in_check(BID, R, C).

% Can a black piece (non-king) reach a square?
% We check black rooks, bishops, queens, knights, pawns

black_piece_can_reach(ID, R2, C2) :-
    placement(black, rook, ID, R1, C1),
    rook_moves(R1, C1, R2, C2),
    not blocked(after, R1, C1, R2, C2),
    not black_piece_at_after(R2, C2).

black_piece_can_reach(ID, R2, C2) :-
    placement(black, bishop, ID, R1, C1),
    bishop_moves(R1, C1, R2, C2),
    not blocked(after, R1, C1, R2, C2),
    not black_piece_at_after(R2, C2).

black_piece_can_reach(ID, R2, C2) :-
    placement(black, queen, ID, R1, C1),
    queen_moves(R1, C1, R2, C2),
    not blocked(after, R1, C1, R2, C2),
    not black_piece_at_after(R2, C2).

black_piece_can_reach(ID, R2, C2) :-
    placement(black, knight, ID, R1, C1),
    knight_moves(R1, C1, R2, C2),
    not black_piece_at_after(R2, C2).

black_piece_can_reach(ID, R2, C2) :-
    placement(black, pawn, ID, R1, C1),
    % black pawn forward move (no capture)
    R2 = R1+1, C2 = C1,
    not occupied(after, R2, C2).

black_piece_can_reach(ID, R2, C2) :-
    placement(black, pawn, ID, R1, C1),
    % black pawn diagonal capture
    R2 = R1+1, (C2 = C1+1 ; C2 = C1-1),
    square(R2, C2),
    occupied(after, R2, C2),
    not black_piece_at_after(R2, C2).

% Check if capturing/interposing leaves black king in check
% Simplified: after black piece BID moves to (TR, TC), does white still threaten bk?
% We use a conservative approach: check if there's a double check (two checkers)
% In double check, only king moves can escape

% Count checking pieces
num_checkers(N) :- N = #count { ID : checking_piece(ID, _, _) }.

double_check :- num_checkers(N), N >= 2.

% If double check, black cannot block or capture (only king moves help)
% If single check, black might be able to block or capture

% After black captures the checker, is the black king still in check?
% We need to check if white has another piece giving check
% This is approximated by checking if there's only one checker and it gets captured

capture_leaves_bk_in_check(BID, CR, CC) :-
    % After BID captures at (CR,CC), is bk still in check?
    % Check if any other white piece (not the captured one) still attacks bk
    checking_piece(OtherID, _, _),
    OtherID != ID,
    checking_piece(ID, CR, CC).

% Actually let's simplify: if there's only one checker and black captures it,
% we need to verify the capturing piece doesn't leave the king exposed
% For now, use a simpler model: black can escape if not double check and can capture/block

% Black has a non-king escape if:
% - not double check AND (can capture checker OR can interpose)
black_has_non_king_escape :-
    not double_check,
    black_can_capture_checker.

black_has_non_king_escape :-
    not double_check,
    black_can_interpose.

% Black has any escape
black_has_escape :- bk_has_king_escape.
black_has_escape :- black_has_non_king_escape.

% 8. Checkmate constraint: reject moves that don't give checkmate
:- not black_king_in_check.
:- black_has_escape.

% 9. Output the mating move
mate_move(white, T, R1, C1, R2, C2) :-
    choose_move(ID, R1, C1, R2, C2),
    placement(white, T, ID, R1, C1).

% Helper: white king threatens after (same as white_threatens for king, already included)
white_king_threatens_after(R2, C2) :-
    new_placement(wk, R1, C1),
    king_moves(R1, C1, R2, C2).

#show mate_move/6.