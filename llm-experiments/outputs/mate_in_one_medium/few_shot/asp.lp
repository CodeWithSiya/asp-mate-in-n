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
% MATE-IN-ONE REASONING LAYER
%=============================================================

% Select exactly one legal white move
1 { chosen_move(ID, R1, C1, R2, C2) : legal_move(ID, R1, C1, R2, C2) } 1.

%-------------------------------------------------------------
% Derive new_placement after the chosen white move
%-------------------------------------------------------------

% The moved piece goes to its new square
new_placement(ID, R2, C2) :- chosen_move(ID, R1, C1, R2, C2).

% All other white pieces stay where they are (not the moved piece)
new_placement(ID, R, C) :-
    placement(white, _, ID, R, C),
    not chosen_move(ID, R, C, _, _),
    % make sure this piece is not the one that moved
    not moved_white_piece(ID).

moved_white_piece(ID) :- chosen_move(ID, _, _, _, _).

% Black pieces remain unless captured by the chosen move
new_placement(ID, R, C) :-
    placement(black, _, ID, R, C),
    not captured_by_white(ID).

% A black piece is captured if white moves to its square
captured_by_white(ID) :-
    chosen_move(_, _, _, R2, C2),
    placement(black, _, ID, R2, C2).

%-------------------------------------------------------------
% Determine piece types for new_placement (needed for white_threatens)
%-------------------------------------------------------------
% piece_type already defined for white pieces above in base fragment

%-------------------------------------------------------------
% Check: black king is in check after white's move
%-------------------------------------------------------------

bk_in_check :-
    placement(black, king, bk, BKR, BKC),
    white_threatens(BKR, BKC).

%-------------------------------------------------------------
% Black king escape analysis
% We need to check if the black king has ANY legal reply
% that gets it out of check.
%-------------------------------------------------------------

% Position of black king
bk_pos(R, C) :- placement(black, king, bk, R, C).

% Squares the black king can potentially move to (king move geometry)
bk_candidate_escape(R2, C2) :-
    bk_pos(R1, C1),
    king_moves(R1, C1, R2, C2),
    square(R2, C2).

% A square is occupied after white's move (for black king escape analysis)
% We use new_placement for this
new_occupied(R, C) :- new_placement(_, R, C).

% Black king cannot move to a square occupied by a black piece (friendly)
new_friendly_black(R, C) :-
    placement(black, _, ID, R, C),
    not captured_by_white(ID).

% After black king moves to (R2,C2), we need to check if white still attacks that square.
% We compute white attacks on the post-move board but with the black king at (R2,C2).
% For this we use white_threatens which already excludes the black king's original square
% via occupied_excluding_bk / blocked_excluding_bk.

% Black king escape: move to (R2,C2) is valid if:
%   1. It's a king_moves destination
%   2. Not occupied by a friendly black piece
%   3. Not attacked by white (using white_threatens which accounts for bk removal)
%   4. The square is on the board

bk_can_escape_to(R2, C2) :-
    bk_candidate_escape(R2, C2),
    not new_friendly_black(R2, C2),
    not white_threatens(R2, C2).

% Black has an escape if there exists at least one escape square
bk_has_escape :- bk_can_escape_to(_, _).

%-------------------------------------------------------------
% Check if any black non-king piece can block or capture the attacker
%-------------------------------------------------------------

% Identify which white piece is giving check (the attacker)
checking_piece(ID, R1, C1) :-
    new_placement(ID, R1, C1),
    placement(white, _, ID, _, _),  % it's a white piece
    bk_pos(BKR, BKC),
    white_piece_attacks_sq(ID, R1, C1, BKR, BKC).

% White piece attacks a square (used to find checking piece)
white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, rook),
    rook_moves(R1, C1, R2, C2),
    not blocked_excluding_bk(R1, C1, R2, C2).

white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, queen),
    queen_moves(R1, C1, R2, C2),
    not blocked_excluding_bk(R1, C1, R2, C2).

white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, bishop),
    bishop_moves(R1, C1, R2, C2),
    not blocked_excluding_bk(R1, C1, R2, C2).

white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, knight),
    knight_moves(R1, C1, R2, C2).

white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1+1,
    square(R2, C2).

white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1-1,
    square(R2, C2).

white_piece_attacks_sq(ID, R1, C1, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, king),
    king_moves(R1, C1, R2, C2).

%-------------------------------------------------------------
% Squares between checker and black king (for interposition)
%-------------------------------------------------------------

% Squares on the line between checker and black king (exclusive)
interpose_square(IR, IC) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    square(IR, IC),
    between_horizontal(CR, CC, IC, BKC),
    CR = BKR, IR = CR.

interpose_square(IR, IC) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    square(IR, IC),
    between_horizontal(CR, BKC, IC, CC),
    CR = BKR, IR = CR.

interpose_square(IR, IC) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    square(IR, IC),
    between_vertical(CC, CR, IR, BKR),
    CC = BKC, IC = CC.

interpose_square(IR, IC) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    square(IR, IC),
    between_vertical(CC, BKR, IR, CR),
    CC = BKC, IC = CC.

interpose_square(IR, IC) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    square(IR, IC),
    between_diagonal(IR, IC, CR, CC, BKR, BKC).

interpose_square(IR, IC) :-
    checking_piece(_, CR, CC),
    bk_pos(BKR, BKC),
    square(IR, IC),
    between_diagonal(IR, IC, BKR, BKC, CR, CC).

%-------------------------------------------------------------
% Black non-king pieces and their possible moves after white moves
%-------------------------------------------------------------

% Black non-king pieces still on board
black_piece_on_board(ID, PT, R, C) :-
    placement(black, PT, ID, R, C),
    PT != king,
    not captured_by_white(ID).

% Occupied squares after white's move (for black's move legality)
% Black pieces still on board + white new placements
after_occupied(R, C) :- new_placement(_, R, C).
after_occupied(R, C) :- black_piece_on_board(_, _, R, C).

% Friendly for black (after white's move)
after_friendly_black(R, C) :- black_piece_on_board(_, _, R, C).

% Blocked for black pieces moving after white's move
% We reuse the blocked/3 concept but for the "after" state
% after_blocked uses after_occupied

after_blocked(R1, C1, R2, C2) :-
    between_horizontal(R1, C1, C_curr, C2),
    after_occupied(R1, C_curr), R2 = R1.
after_blocked(R1, C1, R2, C2) :-
    between_horizontal(R1, C2, C_curr, C1),
    after_occupied(R1, C_curr), R2 = R1.
after_blocked(R1, C1, R2, C2) :-
    between_vertical(C1, R1, R_curr, R2),
    after_occupied(R_curr, C1), C2 = C1.
after_blocked(R1, C1, R2, C2) :-
    between_vertical(C1, R2, R_curr, R1),
    after_occupied(R_curr, C1), C2 = C1.
after_blocked(R1, C1, R2, C2) :-
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2),
    after_occupied(R_curr, C_curr).

% Legal moves for black non-king pieces (rook, bishop, queen, knight, pawn)
black_piece_legal_move(ID, R1, C1, R2, C2) :-
    black_piece_on_board(ID, rook, R1, C1),
    rook_moves(R1, C1, R2, C2),
    not after_blocked(R1, C1, R2, C2),
    not after_friendly_black(R2, C2).

black_piece_legal_move(ID, R1, C1, R2, C2) :-
    black_piece_on_board(ID, bishop, R1, C1),
    bishop_moves(R1, C1, R2, C2),
    not after_blocked(R1, C1, R2, C2),
    not after_friendly_black(R2, C2).

black_piece_legal_move(ID, R1, C1, R2, C2) :-
    black_piece_on_board(ID, queen, R1, C1),
    queen_moves(R1, C1, R2, C2),
    not after_blocked(R1, C1, R2, C2),
    not after_friendly_black(R2, C2).

black_piece_legal_move(ID, R1, C1, R2, C2) :-
    black_piece_on_board(ID, knight, R1, C1),
    knight_moves(R1, C1, R2, C2),
    not after_friendly_black(R2, C2).

% Black pawn forward move
black_piece_legal_move(ID, R1, C1, R2, C2) :-
    black_piece_on_board(ID, pawn, R1, C1),
    R2 = R1+1, C2 = C1,
    square(R2, C2),
    not after_occupied(R2, C2).

% Black pawn diagonal capture
black_piece_legal_move(ID, R1, C1, R2, C2) :-
    black_piece_on_board(ID, pawn, R1, C1),
    R2 = R1+1, C2 = C1+1,
    square(R2, C2),
    after_occupied(R2, C2),
    not after_friendly_black(R2, C2).

black_piece_legal_move(ID, R1, C1, R2, C2) :-
    black_piece_on_board(ID, pawn, R1, C1),
    R2 = R1+1, C2 = C1-1,
    square(R2, C2),
    after_occupied(R2, C2),
    not after_friendly_black(R2, C2).

%-------------------------------------------------------------
% Can a black non-king piece capture the checking piece?
%-------------------------------------------------------------

black_can_capture_checker :-
    checking_piece(CID, CR, CC),
    black_piece_legal_move(_, _, _, CR, CC),
    % Make sure it's not a double check (two checkers means only king move works)
    not double_check.

%-------------------------------------------------------------
% Double check detection (two pieces giving check simultaneously)
%-------------------------------------------------------------

num_checkers(N) :- N = #count { ID : checking_piece(ID, _, _) }.
double_check :- num_checkers(N), N >= 2.

%-------------------------------------------------------------
% Can a black non-king piece interpose (block the check)?
%-------------------------------------------------------------

black_can_interpose :-
    interpose_square(IR, IC),
    black_piece_legal_move(_, _, _, IR, IC),
    not double_check.

%-------------------------------------------------------------
% Black has a defense if:
%   - king can escape, OR
%   - a piece can capture the checker (not double check), OR
%   - a piece can interpose (not double check, not knight/pawn check)
%-------------------------------------------------------------

% For knight or pawn checks, interposition is impossible
% (already handled since interpose_square won't generate squares for knights/pawns
%  as they don't move in lines - between_* won't fire)

black_has_defense :-
    bk_has_escape.

black_has_defense :-
    black_can_capture_checker.

black_has_defense :-
    black_can_interpose.

%-------------------------------------------------------------
% A move is a mating move if:
%   1. It puts the black king in check
%   2. Black has no defense
%-------------------------------------------------------------

is_checkmate :-
    bk_in_check,
    not black_has_defense.

% Constraint: we only want answers where the chosen move is checkmate
:- not is_checkmate.

%-------------------------------------------------------------
% Output the mating move
%-------------------------------------------------------------

mate_move(ID, R1, C1, R2, C2, checkmate) :-
    chosen_move(ID, R1, C1, R2, C2),
    is_checkmate.

#show mate_move/6.