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
% REASONING LAYER: Mate-in-one solver
%---------------------------------------------------------------------------------------------------------------------------

% Select exactly one legal white move
1 { chosen_move(ID, R1, C1, R2, C2) : legal_move(ID, R1, C1, R2, C2), placement(white, _, ID, R1, C1) } 1.

%---------------------------------------------------------------------------------------------------------------------------
% Derive new_placement after the chosen white move
%---------------------------------------------------------------------------------------------------------------------------

% The moved piece goes to its new square
new_placement(ID, R2, C2) :- chosen_move(ID, R1, C1, R2, C2).

% All white pieces that did not move stay in place
new_placement(ID, R, C) :- 
    placement(white, _, ID, R, C),
    not chosen_move(ID, R, C, _, _),
    not chosen_move(ID, _, _, _, _).

% White pieces that did not move stay (handle the case where chosen piece moved away)
new_placement(ID, R, C) :-
    placement(white, _, ID, R, C),
    chosen_move(OtherID, _, _, _, _),
    OtherID != ID.

% Black king stays (unless captured, but black king cannot be captured in a legal game sense;
% however if queen moves to black king square that would be a capture - we keep black king only if not captured)
new_placement(bk, R, C) :-
    placement(black, king, bk, R, C),
    not chosen_move(_, _, _, R, C).

%---------------------------------------------------------------------------------------------------------------------------
% Determine piece types after the move (for attack computation)
%---------------------------------------------------------------------------------------------------------------------------

% Which piece is at each square after the move, and what type
after_piece(ID, Piece, R, C) :-
    placement(white, Piece, ID, _, _),
    new_placement(ID, R, C).

after_piece(bk, king, R, C) :-
    new_placement(bk, R, C).

%---------------------------------------------------------------------------------------------------------------------------
% Compute blocked_after: blocking on the after-state board
%---------------------------------------------------------------------------------------------------------------------------

blocked_after(R1, C1, R2, C2) :-
    between_horizontal(R1, C1, C_curr, C2),
    occupied(after, R1, C_curr), R2 = R1.
blocked_after(R1, C1, R2, C2) :-
    between_horizontal(R1, C2, C_curr, C1),
    occupied(after, R1, C_curr), R2 = R1.

blocked_after(R1, C1, R2, C2) :-
    between_vertical(C1, R1, R_curr, R2),
    occupied(after, R_curr, C1), C2 = C1.
blocked_after(R1, C1, R2, C2) :-
    between_vertical(C1, R2, R_curr, R1),
    occupied(after, R_curr, C1), C2 = C1.

blocked_after(R1, C1, R2, C2) :-
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2),
    occupied(after, R_curr, C_curr).

%---------------------------------------------------------------------------------------------------------------------------
% White attacks after the move (squares white pieces attack on the after-state board)
%---------------------------------------------------------------------------------------------------------------------------

% White queen attacks
white_attacks_after(R2, C2) :-
    after_piece(wq, queen, R1, C1),
    queen_moves(R1, C1, R2, C2),
    not blocked_after(R1, C1, R2, C2).

% White king attacks
white_attacks_after(R2, C2) :-
    after_piece(wk, king, R1, C1),
    king_moves(R1, C1, R2, C2).

%---------------------------------------------------------------------------------------------------------------------------
% Check: is the black king in check after white's move?
%---------------------------------------------------------------------------------------------------------------------------

black_king_in_check :-
    new_placement(bk, BKR, BKC),
    white_attacks_after(BKR, BKC).

%---------------------------------------------------------------------------------------------------------------------------
% Black king escape analysis
% For each possible black king move after white's chosen move, determine if it escapes check
%---------------------------------------------------------------------------------------------------------------------------

% Black king's current position after white moves (same as before unless captured, which can't happen for king)
black_king_pos(R, C) :- new_placement(bk, R, C).

% A square is occupied by white after the move
white_occupied_after(R, C) :-
    after_piece(ID, _, R, C),
    placement(white, _, ID, _, _).

% Black king candidate moves: adjacent squares
bk_candidate(R1, C1, R2, C2) :-
    black_king_pos(R1, C1),
    king_moves(R1, C1, R2, C2).

% A candidate black king move is blocked if the destination is occupied by a black piece
% (only black piece is the king itself, so no blocking by own pieces other than king)
% The destination must not be occupied by a black piece (there are none other than king)

% After black king moves to (R2,C2), compute what the board looks like:
% White pieces remain as in after-state, black king moves to (R2,C2)

% White attacks a square given the after-state board but with black king at (R2,C2)
% We need to check if white attacks (R2,C2) after black king moves there
% The board after black king moves: white pieces same, black king at (R2,C2), old black king square empty

% Blocked status for white pieces attacking (R2,C2) when black king is at (BKR2,BKC2)
% We need to recompute blocking considering black king moved from (BKR1,BKC1) to (BKR2,BKC2)

% Squares occupied after black king moves to (BKR2, BKC2)
% White pieces: same as after white's move
% Black king: now at (BKR2, BKC2)

occupied_after_bk_move(R, C, BKR2, BKC2) :-
    white_occupied_after(R, C),
    bk_candidate(_, _, BKR2, BKC2).

occupied_after_bk_move(BKR2, BKC2, BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2).

% Blocking for white queen when black king moves to (BKR2,BKC2)
blocked_after_bk(R1, C1, R2, C2, BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    between_horizontal(R1, C1, C_curr, C2),
    occupied_after_bk_move(R1, C_curr, BKR2, BKC2),
    R2 = R1.
blocked_after_bk(R1, C1, R2, C2, BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    between_horizontal(R1, C2, C_curr, C1),
    occupied_after_bk_move(R1, C_curr, BKR2, BKC2),
    R2 = R1.

blocked_after_bk(R1, C1, R2, C2, BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    between_vertical(C1, R1, R_curr, R2),
    occupied_after_bk_move(R_curr, C1, BKR2, BKC2),
    C2 = C1.
blocked_after_bk(R1, C1, R2, C2, BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    between_vertical(C1, R2, R_curr, R1),
    occupied_after_bk_move(R_curr, C1, BKR2, BKC2),
    C2 = C1.

blocked_after_bk(R1, C1, R2, C2, BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2),
    occupied_after_bk_move(R_curr, C_curr, BKR2, BKC2).

% White queen attacks (BKR2,BKC2) after black king moves there
white_attacks_after_bk_move(BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    after_piece(wq, queen, QR, QC),
    queen_moves(QR, QC, BKR2, BKC2),
    not blocked_after_bk(QR, QC, BKR2, BKC2, BKR2, BKC2).

% White king attacks (BKR2,BKC2) after black king moves there
white_attacks_after_bk_move(BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    after_piece(wk, king, WKR, WKC),
    king_moves(WKR, WKC, BKR2, BKC2).

% A black king escape move is legal if:
% 1. destination is not occupied by a white piece (can capture white piece only if it's not protected,
%    but for simplicity: destination not occupied by white non-capturable piece)
%    Actually black king CAN capture white pieces if they are not protected.
%    The destination must not be attacked by white after the capture.
%    If black king captures a white piece at (R2,C2), that white piece is gone.

% Let's handle capture: if black king moves to a white-occupied square, that white piece is captured.
% After capture, we need to check if (R2,C2) is still attacked by remaining white pieces.

% Squares occupied after black king captures white piece at (BKR2,BKC2)
occupied_after_bk_capture(R, C, BKR2, BKC2) :-
    white_occupied_after(R, C),
    bk_candidate(_, _, BKR2, BKC2),
    (R, C) != (BKR2, BKC2).  % the captured piece is removed

occupied_after_bk_capture(BKR2, BKC2, BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2).

% Blocking for white queen when black king captures at (BKR2,BKC2)
blocked_after_bk_capture(R1, C1, R2, C2, BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    white_occupied_after(BKR2, BKC2),  % there is a white piece to capture
    between_horizontal(R1, C1, C_curr, C2),
    occupied_after_bk_capture(R1, C_curr, BKR2, BKC2),
    R2 = R1.
blocked_after_bk_capture(R1, C1, R2, C2, BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    white_occupied_after(BKR2, BKC2),
    between_horizontal(R1, C2, C_curr, C1),
    occupied_after_bk_capture(R1, C_curr, BKR2, BKC2),
    R2 = R1.

blocked_after_bk_capture(R1, C1, R2, C2, BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    white_occupied_after(BKR2, BKC2),
    between_vertical(C1, R1, R_curr, R2),
    occupied_after_bk_capture(R_curr, C1, BKR2, BKC2),
    C2 = C1.
blocked_after_bk_capture(R1, C1, R2, C2, BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    white_occupied_after(BKR2, BKC2),
    between_vertical(C1, R2, R_curr, R1),
    occupied_after_bk_capture(R_curr, C1, BKR2, BKC2),
    C2 = C1.

blocked_after_bk_capture(R1, C1, R2, C2, BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    white_occupied_after(BKR2, BKC2),
    between_diagonal(R_curr, C_curr, R1, C1, R2, C2),
    occupied_after_bk_capture(R_curr, C_curr, BKR2, BKC2).

% White queen attacks (BKR2,BKC2) after black king captures white piece there
white_attacks_after_bk_capture(BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    white_occupied_after(BKR2, BKC2),
    after_piece(wq, queen, QR, QC),
    (QR, QC) != (BKR2, BKC2),  % the queen is not the one being captured
    queen_moves(QR, QC, BKR2, BKC2),
    not blocked_after_bk_capture(QR, QC, BKR2, BKC2, BKR2, BKC2).

% White king attacks (BKR2,BKC2) after black king captures white piece there
white_attacks_after_bk_capture(BKR2, BKC2) :-
    bk_candidate(_, _, BKR2, BKC2),
    white_occupied_after(BKR2, BKC2),
    after_piece(wk, king, WKR, WKC),
    (WKR, WKC) != (BKR2, BKC2),
    king_moves(WKR, WKC, BKR2, BKC2).

%---------------------------------------------------------------------------------------------------------------------------
% Determine if a black king candidate move is a valid escape
%---------------------------------------------------------------------------------------------------------------------------

% A move to an empty square (not white-occupied) is an escape if white does not attack that square after bk moves
bk_escape(R1, C1, R2, C2) :-
    bk_candidate(R1, C1, R2, C2),
    not white_occupied_after(R2, C2),
    not white_attacks_after_bk_move(R2, C2).

% A move to a white-occupied square (capture) is an escape if white does not attack that square after capture
bk_escape(R1, C1, R2, C2) :-
    bk_candidate(R1, C1, R2, C2),
    white_occupied_after(R2, C2),
    not white_attacks_after_bk_capture(R2, C2).

%---------------------------------------------------------------------------------------------------------------------------
% Checkmate: black king is in check and has no escape
%---------------------------------------------------------------------------------------------------------------------------

checkmate :-
    black_king_in_check,
    not bk_escape(_, _, _, _).

%---------------------------------------------------------------------------------------------------------------------------
% Constraint: only keep chosen moves that result in checkmate
%---------------------------------------------------------------------------------------------------------------------------

:- not checkmate.

%---------------------------------------------------------------------------------------------------------------------------
% Output the mating move
%---------------------------------------------------------------------------------------------------------------------------

mate_move(ID, R1, C1, R2, C2, checkmate) :-
    chosen_move(ID, R1, C1, R2, C2),
    checkmate.

#show mate_move/6.