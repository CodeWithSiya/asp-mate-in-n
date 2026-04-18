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
% EXTENSION: Mate-in-one reasoning
%=============================================================

% 1. Choose exactly one legal white move
{ choose_move(ID, R1, C1, R2, C2) : legal_move(ID, R1, C1, R2, C2), placement(white, _, ID, R1, C1) } = 1.

% 2. Derive new_placement after the chosen move
% The moved piece goes to its new square
new_placement(ID, R2, C2) :- choose_move(ID, R1, C1, R2, C2).

% All other white pieces stay where they are
new_placement(ID, R, C) :- 
    placement(white, _, ID, R, C),
    not choose_move(ID, R, C, _, _),
    not choose_move(ID, _, _, R, C).

% White piece that moved: don't keep it at old square (handled above by the not choose_move condition)

% Black pieces stay unless captured
new_placement(ID, R, C) :- 
    placement(black, _, ID, R, C),
    not choose_move(_, _, _, R, C).  % not captured by white's move

% 3. Check detection: black king is in check after white's move
% The black king's position after white moves (it hasn't moved yet)
bk_pos(R, C) :- placement(black, king, bk, R, C).

% White threatens the black king's square => check
black_king_in_check :- 
    bk_pos(R, C),
    white_threatens(R, C).

% 4. Escape detection for black king

% Squares occupied by white pieces after the move (black king cannot move there)
white_occupied_after(R, C) :- new_placement(ID, R, C), placement(white, _, ID, _, _).

% Black king can potentially escape to a square if:
% - it's a valid king move
% - not occupied by a black piece (friendly)
% - not threatened by white after king moves there
% - not occupied by white (unless capturing, but white pieces block)

% Black friendly pieces (non-king) after white's move
black_friendly_after(R, C) :- 
    new_placement(ID, R, C), 
    placement(black, _, ID, _, _),
    ID != bk.

% After black king moves to (R2,C2), we need to check if white still threatens that square
% We use white_threatens which is computed excluding the black king already

% Black king escape: king moves to (R2,C2)
bk_can_escape_to(R2, C2) :-
    bk_pos(R1, C1),
    king_moves(R1, C1, R2, C2),
    not black_friendly_after(R2, C2),   % not blocked by own piece
    not white_occupied_after(R2, C2),   % can't move to white-occupied square unless capturing... 
    not white_threatens(R2, C2).        % not threatened by white

% Actually black king CAN capture a white piece if it's not protected
% white_occupied_after includes the moved piece; if black king captures it, 
% we need to check if that square is still threatened by OTHER white pieces
% Let's handle capture separately:

% White pieces that are NOT the chosen moved piece, at their positions after move
white_piece_after_excl(ID2, R, C) :-
    new_placement(ID2, R, C),
    placement(white, _, ID2, _, _).

% Is square (R2,C2) threatened by white pieces other than the one that might be captured there?
% We need white_threatens_excl_capture for the square the king moves to
% This is complex; let's use white_threatens but note it already excludes bk position

% For king capturing a white piece at (R2,C2):
% The square must not be defended by another white piece
% white_threatens already computes threats from all white pieces in new_placement
% If black king captures the white piece at (R2,C2), that piece is gone
% So we need: is (R2,C2) threatened by white pieces OTHER than the one at (R2,C2)?

% Compute white threatens excluding a specific square (for capture check)
white_threatens_excl(ExR, ExC, R2, C2) :-
    new_placement(ID, R1, C1),
    placement(white, king, ID, _, _),
    king_moves(R1, C1, R2, C2),
    (R1, C1) != (ExR, ExC).

white_threatens_excl(ExR, ExC, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, rook),
    rook_moves(R1, C1, R2, C2),
    (R1, C1) != (ExR, ExC),
    not blocked_excl_bk_and_sq(ExR, ExC, R1, C1, R2, C2).

white_threatens_excl(ExR, ExC, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, bishop),
    bishop_moves(R1, C1, R2, C2),
    (R1, C1) != (ExR, ExC),
    not blocked_excl_bk_and_sq(ExR, ExC, R1, C1, R2, C2).

white_threatens_excl(ExR, ExC, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, queen),
    queen_moves(R1, C1, R2, C2),
    (R1, C1) != (ExR, ExC),
    not blocked_excl_bk_and_sq(ExR, ExC, R1, C1, R2, C2),
    not new_placement(ID, R2, C2).

white_threatens_excl(ExR, ExC, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, knight),
    knight_moves(R1, C1, R2, C2),
    (R1, C1) != (ExR, ExC).

white_threatens_excl(ExR, ExC, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1+1,
    square(R2, C2),
    (R1, C1) != (ExR, ExC).

white_threatens_excl(ExR, ExC, R2, C2) :-
    new_placement(ID, R1, C1),
    piece_type(white, ID, pawn),
    R2 = R1-1, C2 = C1-1,
    square(R2, C2),
    (R1, C1) != (ExR, ExC).

% blocked excluding bk and a specific square (for when a piece is captured)
occupied_excl_bk_and_sq(ExR, ExC, R, C) :-
    occupied_excluding_bk(R, C),
    (R, C) != (ExR, ExC).

blocked_excl_bk_and_sq(ExR, ExC, R1, C1, R2, C2) :-
    between_horizontal(R1, C1, C_curr, C2),
    occupied_excl_bk_and_sq(ExR, ExC, R1, C_curr), R2 = R1.
blocked_excl_bk_and_sq(ExR, ExC, R1, C1, R2, C2) :-
    between_horizontal(R1, C2, C_curr, C1),
    occupied_excl_bk_and_sq(ExR, ExC, R1, C_curr), R2 = R1.
blocked_excl_bk_and_sq(ExR, ExC, R1, C1, R2, C2) :-
    between_vertical(C1, R1, R_curr, R2),
    occupied_excl_bk_and_sq(ExR, ExC, R_curr, C1), C2 = C1.
blocked_excl_bk_and_sq(ExR, ExC, R1, C1, R2, C2) :-
    between_vertical(C1, R2, R_curr, R1),
    occupied_excl_bk_and_sq(ExR, ExC, R_curr, C1), C2 = C1.
blocked_excl_bk_and_sq(ExR, ExC, R1, C1, R2, C2) :-
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2),
    occupied_excl_bk_and_sq(ExR, ExC, R_curr, C_curr).

% Black king can escape by capturing a white piece at (R2,C2) if that square is not defended
bk_can_escape_to(R2, C2) :-
    bk_pos(R1, C1),
    king_moves(R1, C1, R2, C2),
    white_occupied_after(R2, C2),           % there's a white piece to capture
    not white_threatens_excl(R2, C2, R2, C2). % but it's not defended by other white pieces

% 5. Checkmate: black king is in check and has no escape
checkmate :-
    black_king_in_check,
    not bk_can_escape_to(_, _).

% 6. Reject any chosen move that doesn't deliver checkmate
:- not checkmate.

% 7. Output the mating move
mate_move(white, T, R1, C1, R2, C2) :-
    choose_move(ID, R1, C1, R2, C2),
    placement(white, T, ID, R1, C1).

#show mate_move/6.